"""Per-frame soccer tracking pipeline (everything after the detector):

    person detections -> camera motion -> pitch detection and the tracked pitch model -> outside-field
    filter (foot point) -> local tracker (BoT-SORT) -> appearance (OSNet + part colours) -> kit votes +
    position cues -> pitch coordinates -> GLOBAL IDENTITY MANAGER (identity, team and role) ->
    persistent soccer observations; ball candidates -> one tracked match ball

Pitch modes. AUTOMATIC: the image-plane detector finds painted boundary lines (broadcast footage) or
the tracked pitch model is confident; nothing below changes. ROBUST: no painted boundary line is
found (low, wide-angle amateur cameras with faded lines), so the playable area is the pitch model's
temporal prediction when there is one, else the grass region held with a margin while new people
stay provisional; far-away people are looked for with extra tiles (pipeline.far_tiles) and kept only
when confirmed. ASSIST: the confidence has stayed low for a sustained section and no model exists;
the section is reported once as "a pitch section needs confirmation" with a representative frame and
the landmarks worth marking, which the next run turns into an anchored model.
"""
import math
from dataclasses import dataclass

import numpy as np

from . import appearance as app
from .ball import BallTracker
from .camera import CameraMotion, CameraState, Keyframes
from .geometry import foot_point, iou
from .identity import PLAYER_HEIGHT, IdentityManager, Options, Stab
from . import midfield
from .pitch import (FieldFilter, PitchDetector, assess_confidence, calibrated_model, find_circle, halfway_candidate, nearest_boundary,
                    suggest_landmarks, warp_model)
from .pitch_model import HIGH, LOW_LEVEL
from .registration import Anchor
from .relevance import REJECT_TEXT, assess
from .goals import GoalMemory, position_cues
from .teams import TeamModel, fit_team_model, kit_vote, with_keepers, with_referee

CLASSES = {1: 'goalkeeper', 2: 'player', 3: 'referee'}
MIN_VOTE_QUALITY = .2
ASSIST_AFTER = 3.0        # seconds of LOW pitch confidence without a model before a section asks for confirmation
ASSIST_GAP = 3.0          # sections closer than this are one issue
FAR_MIN_PX = 30           # people expected smaller than this at the far edge of the pitch get the far-tile pass
FAR_CONFIRM = 3           # consecutive frames a far-tile detection must repeat before it counts


@dataclass
class Config:
    field_filter: FieldFilter
    options: Options
    boundaries: list
    calibrations: list
    debug: bool = True


def config_from(settings):
    margin = float(settings.get('touchlineMargin', .35))
    field = FieldFilter.with_margin(margin) if settings.get('fieldFilter', True) else FieldFilter(enabled=False)
    options = Options(max_per_team=int(settings.get('maxPerTeam', 11)))
    return Config(field, options, settings.get('boundaries', []), settings.get('calibrations', []), bool(settings.get('debug', True)))


def _occluded(box, others, aspect):
    for o in others:
        if o is box:
            continue
        if iou(o, box) > .02 or math.hypot((o[0]+o[2]/2-box[0]-box[2]/2)*aspect, o[1]+o[3]/2-box[1]-box[3]/2) < .6*(o[2]+box[2])/2*aspect:
            return True
    return False


class SoccerTracker:
    def __init__(self, width, height, config, local_tracker, embedder=None, saved=None, fps=25.0):
        self.width, self.height = width, height
        self.ball = BallTracker(width, height, fps)
        self.aspect = width/height
        self.config = config
        self.local = local_tracker
        self.embedder = embedder
        self.camera = CameraMotion()
        self.state = CameraState(width, height)
        self.keyframes = Keyframes(config.boundaries, config.calibrations, width, height)
        self.pitch_detector = PitchDetector()
        self.goals = GoalMemory()
        self.analysed = self.pitch = None
        self.carried_motion, self.carried_reliable = np.eye(3), True
        self.size = None
        self.ids = IdentityManager(config.options, (saved or {}).get('registry'))
        self.model = TeamModel.from_json((saved or {}).get('teams'))
        self.fit_at = -math.inf
        self.last_tick = -math.inf
        self.last_time = None
        self.previous_boxes = []
        self.rejected_counts = {}
        self.events = []
        self.issues = []
        self.cuts = 0
        self.last_ball = None   # the previous frame's ball position, a role cue for everyone
        self.pitch_events_logged = 0
        self.frame_log = []     # per frame: camera motion (and how it was measured) and the live pitch model, for the offline passes
        self.frame_index = -1   # index of the current frame in this run
        self.anchors = []       # registration.Anchor: every validated calibration of this run, for the offline passes
        self.circle = None      # centre-circle paint found on the last evidence tick (pitch.find_circle), or None
        self.midfield = None    # the last automatic midfield registration attempt: {'time', 'taken', 'reason'|'observability'}
        self.registration = None   # filled by the pipeline's offline passes
        self.low_since = None   # start of the current LOW-confidence section without a model
        self.section = None     # its best representative tick so far
        self.sections = []      # closed LOW sections [{'start', 'end', 'at', 'suggested'}]
        self.far_pending = []   # far-tile detections waiting for confirmation: [box, hits, last time]
        self.mode = 'automatic'

    def _scale(self, boxes):
        """Expected person height (normalized) at a foot row: the perspective size model, else the median
        person on this frame."""
        if self.size is not None and self.size.reliable:
            return lambda y: max(1e-3, self.size.expected(y))
        heights = sorted(b[3] for b in boxes)
        return (lambda y: heights[len(heights)//2]) if heights else None

    def _live_boxes(self, time):
        return [tr.last_box for tr in self.ids.tracks.values() if tr.state in ('confirmed', 'uncertain') and time-tr.last_time < 1]

    def _confirm_far(self, box, live, time):
        """A far-tile detection counts when it continues a tracked person or repeats on consecutive frames."""
        if any(iou(box, b) > .3 for b in live):
            return True
        self.far_pending = [p for p in self.far_pending if time-p[2] <= .5]
        for pending in self.far_pending:
            if iou(box, pending[0]) > .5:
                pending[0], pending[1], pending[2] = box, pending[1]+1, time
                return pending[1] >= FAR_CONFIRM
        self.far_pending.append([box, 1, time])
        return False

    def far_band(self):
        """Image rows (pixels) where people are expected to be too small for the whole-frame detector,
        for an extra tiled pass; None in automatic mode or when nothing there is that small."""
        pitch = self.pitch
        if self.mode == 'automatic' or pitch is None or not pitch.reliable or len(pitch.polygon) < 3 or self.size is None or not self.size.reliable:
            return None
        far_y = max(0.0, min(p[1] for p in pitch.polygon))
        if self.size.expected(far_y+.02)*self.height >= FAR_MIN_PX:
            return None
        y = far_y
        while y < far_y+.35 and self.size.expected(y)*self.height < 2*FAR_MIN_PX:
            y += .01
        return int(max(0.0, far_y-.04)*self.height), int(min(1.0, y)*self.height)

    def step(self, frame, time, rows, ball_rows=None):
        """rows: (N, 6) person detections [x1, y1, x2, y2, score, class id] in pixels, or (N, 7) with a
        last column set for provisional far-tile detections; ball_rows: (M, 6) ball candidates."""
        W, H = self.width, self.height
        rows = np.asarray(rows, np.float32)
        columns = 7 if rows.ndim == 2 and rows.shape[1] == 7 else 6
        rows = rows.reshape(-1, columns)
        flags = rows[:, 6] > 0 if columns == 7 else np.zeros(len(rows), bool)
        rows = rows[:, :6]
        dt = max(0.0, time-self.last_time) if self.last_time is not None else 0.0
        self.last_time = time
        self.frame_index += 1
        live = self._live_boxes(time)
        dets, unconfirmed = [], []
        for i, row in enumerate(rows):
            x1, y1, x2, y2 = np.clip(row[:4], [0, 0, 0, 0], [W, H, W, H])
            if x2-x1 < 2 or y2-y1 < 2:
                continue
            box = [float(x1/W), float(y1/H), float((x2-x1)/W), float((y2-y1)/H)]
            if flags[i] and not self._confirm_far(box, live, time):
                unconfirmed.append({'box': box, 'score': float(row[4]), 'reason': 'unconfirmed-far', 'detail': 'seen by the far-tile pass; not yet repeated or tracked'})
                continue
            dets.append({'box': box, 'score': float(row[4]), 'cls': CLASSES.get(int(row[5]), 'player'), 'row': i})
        # 1. Camera motion: a pan is not player movement.
        M, reliable, cut = self.camera.estimate(frame, self.previous_boxes)
        ended = []
        if cut:
            self.cuts += 1
            ended += self.local.reset()
            self.pitch_detector.reset()
            self.size = None
        self.state.update(M, reliable, cut)
        self.keyframes.update(time, M, reliable, cut, self.camera.homography, self.camera.quality)
        model = self.keyframes.pitch
        if self.keyframes.anchored is not None and model.H is not None:
            # A validated landmark calibration: an anchor the offline passes carry both ways.
            self.anchors.append(Anchor(self.frame_index, time, model.H.copy(), model.observability, 'landmarks', dict(model.details)))
        tick = cut or time-self.last_tick >= self.config.options.tick-1e-6
        if tick:
            self.last_tick = time
        # 2. Playable field: the image-plane detection on every evidence tick (carried with the camera
        # motion in between), scored for confidence; the tracked pitch model, anchored by calibration
        # keyframes and confirmed by painted lines, supplies the region whenever it is trusted.
        motion = self.keyframes.normalized_motion(M)
        if tick or self.analysed is None:
            self.analysed = self.pitch_detector.analyze(frame, time, motion @ self.carried_motion, reliable and self.carried_reliable, cut)
            self.carried_motion, self.carried_reliable = np.eye(3), True
            auto_conf, auto_evidence = assess_confidence(self.analysed)
            self.analysed.confidence, self.analysed.evidence = auto_conf, auto_evidence
            # Centre-circle paint (an ellipse around the halfway line) is calibration evidence of its own: with
            # the halfway line it constrains the geometry where the touchlines alone cannot.
            self.circle, circle_line = find_circle(self.pitch_detector.masks, self.analysed)
            if model.H is not None:
                circle_px = [(x*W, y*H) for x, y in self.circle['points']] if self.circle else None
                model.refine(self.analysed.markings, time, [foot_point(b) for b in live], circle_px)
                if model.state == 'refined':
                    self.anchors.append(Anchor(self.frame_index, time, model.H.copy(), model.observability, 'paint', dict(model.details)))
            elif self.circle is not None and auto_conf < HIGH:
                # No model to carry and no confident playable region from the paint: the centre circle, the
                # halfway line and a touchline can anchor one.
                self._midfield_anchor(model, circle_line, time)
            pitch = self.analysed
        else:
            self.carried_motion = motion @ self.carried_motion
            self.carried_reliable &= reliable
            pitch = warp_model(self.pitch, motion, time, reliable, cut)
        pitch = self._decide_pitch(pitch, model, time, tick)
        self.pitch = pitch
        for when, kind, text in model.events[self.pitch_events_logged:]:
            self.events.append({'time': round(when, 3), 'kind': 'pitch', 'track': None, 'playerId': None, 'message': f'PITCH {kind.upper()}\n{text}'})
        self.pitch_events_logged = len(model.events)
        if self.size is not None and not cut:
            if reliable:
                Mn = self.keyframes.normalized_motion(M)
                self.size = self.size.moved(float(np.sqrt(abs(np.linalg.det(Mn[:2, :2])))), float(Mn[1, 2]))
            else:
                self.size = None
        # 3. Outside-field filter on the foot point.
        accepted, continuation, rejected, self.size = assess(dets, pitch, self.config.field_filter, self.size, self.keyframes.boundary(), live, H)
        continuation = [d for d in continuation if any(iou(d['box'], b) > .3 for b in live)]
        rejected += unconfirmed
        for r in rejected:
            self.rejected_counts[r['reason']] = self.rejected_counts.get(r['reason'], 0)+1
        tracked = accepted+continuation
        # 4. Learned appearance for association and re-identification.
        xyxy = [[d['box'][0]*W, d['box'][1]*H, (d['box'][0]+d['box'][2])*W, (d['box'][1]+d['box'][3])*H] for d in tracked]
        embeddings = self.embedder(frame, xyxy) if self.embedder is not None and tracked else [None]*len(tracked)
        local_rows = np.array([b+[d['score'], int(rows[d['row'], 5])] for b, d in zip(xyxy, tracked)], np.float32).reshape(-1, 6)
        feats = np.asarray(embeddings, np.float32) if self.embedder is not None and tracked else None
        matches, gone = self.local.update(local_rows, frame, feats, M if reliable else None)
        ended += gone
        # 5. Part colour descriptors (only on evidence ticks), team model and kit votes.
        boxes = [d['box'] for d in dets]
        boxes_px = [[b[0]*W, b[1]*H, (b[0]+b[2])*W, (b[1]+b[3])*H] for b in boxes]
        grass = app.grass_reference(frame) if tick else None
        descriptors = {}
        if tick:
            for local_id, index in matches:
                descriptors[local_id] = app.describe(frame, xyxy[index], boxes_px, embeddings[index], grass)
        if tick and (not self.model.ready or time-self.fit_at >= 1 or time < self.fit_at):
            current = {lid: (descriptors.get(lid), tracked[i]['cls'], tracked[i]['zone']) for lid, i in matches}
            model_ = fit_team_model(self.ids.team_samples(current, time), self.model)
            self.model = with_keepers(with_referee(model_, self.ids.referee_samples(model_)), self.ids.keeper_samples())
            self.fit_at = time
        calibrated = self.keyframes.H is not None
        samples = []
        for local_id, index in matches:
            d = tracked[index]
            pitch_point = self.keyframes.to_pitch(d['box'])
            sx, sy, sh = self.state.stabilize(d['box'])
            sample = {'id': local_id, 'box': d['box'], 'score': d['score'], 'cls': d['cls'], 'zone': d['zone'],
                      'occluded': _occluded(d['box'], boxes, self.aspect), 'stab': Stab(sx, sy, sh)}
            if pitch_point is not None:
                sample['pitch'] = pitch_point
            descriptor = descriptors.get(local_id)
            if descriptor is not None:
                sample['descriptor'] = descriptor
                if descriptor.quality >= MIN_VOTE_QUALITY:
                    sample['vote'] = kit_vote(self.model, descriptor.jersey, descriptor.shorts)
            samples.append(sample)
        # Where people stand: near a goal (and which), deepest towards a goal, isolated (role evidence).
        if tick:
            self.goals.update(pitch.goals, self.state.stabilize_point, self.state.segment, time)
            cues = position_cues(samples, self.goals, self.aspect)
            edges = pitch.reliable and len(pitch.polygon) >= 3
            for sample in samples:
                sample['cues'] = cues[sample['id']]
                edge = nearest_boundary(pitch, foot_point(sample['box'])) if edges else None
                if edge is not None and edge['side'] == pitch.vanishing:
                    edge = None   # the touchlines converge there: a vanishing point, not a pitch edge anyone can be near
                if edge is not None:
                    # The nearest pitch edge: how far beyond it (metres, negative inside) and which way it runs.
                    dx, dy = (edge['b'][0]-edge['a'][0])*self.aspect, edge['b'][1]-edge['a'][1]
                    length = math.hypot(dx, dy) or 1.0
                    sample['cues']['edge'] = {'side': edge['side'], 'metres': round(edge['distance']/max(sample['box'][3], 1e-3)*PLAYER_HEIGHT, 2),
                                              'dir': (dx/length, dy/length)}
        # 6. Global identity manager.
        result = self.ids.update({'time': time, 'tick': tick, 'samples': samples, 'ended': [{'id': i} for i in ended],
                                  'model': self.model, 'pitch_reliable': pitch.reliable or self.keyframes.boundary() is not None,
                                  'pitch_provisional': pitch.mode != 'automatic' and pitch.level == LOW_LEVEL,
                                  'filter_enabled': self.config.field_filter.enabled, 'aspect': self.aspect,
                                  'segment': self.state.segment, 'drift': self.state.drift,
                                  'ball': self.last_ball if self.last_ball is not None and self.last_ball['segment'] == self.state.segment else None})
        for ghost in result['drop']:
            self.local.forget(ghost)
        # 7. The single match ball (its own tracker, separate from people).
        people = [(p['track'], p['box']) for p in result['people'] if p['state'] != 'rejected']
        people += [(None, b) for b in boxes if not any(iou(b, q) > .5 for _, q in people)]
        ball, ball_candidates = self.ball.step(frame, time, ball_rows if ball_rows is not None else np.zeros((0, 6)), M, reliable, cut,
                                               pitch, people, self._scale(boxes), self.state.stabilize_point, self.state.segment,
                                               self.keyframes.to_pitch_point if calibrated else None, model.confidence if calibrated else 0.0)
        if ball['state'] in ('TRACKED', 'MISSING') and ball['confidence'] >= .4:
            cx, cy = ball['center']
            self.last_ball = {'stab': self.state.stabilize_point(cx, min(1.0, cy+ball['box'][3]/2)), 'segment': self.state.segment,
                              'pitch': tuple(ball['pitch']) if ball.get('pitch') else None}
        else:
            self.last_ball = None
        self.events += result['events']+self.ball.events
        self.ball.events = []
        self.issues += result['issues']
        self.previous_boxes = boxes
        carried = self.camera.homography if self.camera.homography is not None and reliable else M
        self.frame_log.append({'t': time, 'dt': dt, 'M': carried if reliable else None, 'reliable': reliable, 'cut': cut, 'quality': self.camera.quality,
                               'H': model.H.copy() if calibrated else None, 'confidence': model.confidence if calibrated else 0.0})
        boundary = self.keyframes.boundary()
        live_calibration = model.to_json() | {'anchored': model.H is not None, 'cameraMotion': self.camera.quality, 'midfield': self.midfield,
                                              'circle': {'coverage': self.circle['coverage'], 'inliers': self.circle['inliers']} if self.circle else None}
        return {'people': result['people'], 'ball': ball, 'ballCandidates': ball_candidates,
                'rejected': [{'box': [round(v, 4) for v in r['box']], 'reason': r['reason'], 'text': REJECT_TEXT[r['reason']],
                              **({'detail': r['detail'][:90]} if self.config.debug else {})} for r in rejected],
                'pitch': pitch.to_json() if (tick or cut) else None,
                'boundary': None if boundary is None else [[round(x, 4), round(y, 4)] for x, y in boundary],
                'cameraReliable': reliable, 'cut': cut, 'segment': self.state.segment, 'calibrated': calibrated,
                'pitchCoordinates': ('calibration' if model.level == 'HIGH' else 'predicted') if calibrated else None,
                'pitchMode': pitch.mode, 'pitchConfidence': round(float(pitch.confidence), 3),
                'pitchModel': live_calibration if (tick or cut or self.config.debug) else None}, M, reliable

    def _midfield_anchor(self, model, halfway, time):
        """Automatic registration from the centre circle (an ellipse), the halfway line and a touchline
        (soccer/midfield.py), validated against the paint and scored for observability before it anchors
        the pitch model; a refused attempt is reported once per reason."""
        W, H = self.width, self.height
        ellipse = midfield.ellipse_from_points([(x*W, y*H) for x, y in self.circle['points']])
        if ellipse is None:
            return
        touchlines = midfield.touchline_candidates(self.analysed, ellipse, W, H)
        result = midfield.solve(self.circle, halfway, touchlines, W, H, self.keyframes.size, self.pitch_detector.masks)
        if 'H' not in result:
            if self.midfield is None or self.midfield.get('reason') != result['reason']:
                model.events.append((time, 'midfield', f'centre circle seen ({self.circle["coverage"]:.0%} of the arc) but not used as an anchor: {result["reason"]}'))
            self.midfield = {'time': time, 'taken': False, 'reason': result['reason']}
            return
        taken = model.anchor(np.linalg.inv(result['H']), time, note='centre circle, halfway line and touchline', observability=result['observability'],
                             details=result['details'], source='auto-midfield')
        self.midfield = {'time': time, 'taken': bool(taken), 'observability': round(result['observability'], 3), 'rmsPixels': round(result['rms_px'], 1),
                         'evidence': result['details'].get('evidence')}
        if taken:
            self.anchors.append(Anchor(self.frame_index, time, model.H.copy(), model.observability, 'auto-midfield', dict(model.details)))

    def _decide_pitch(self, pitch, model, time, tick):
        """Which playable region this frame uses, its mode and confidence; LOW sections are recorded."""
        auto_conf, auto_evidence = pitch.confidence, list(pitch.evidence)
        # The tracked model supplies the region when it is trusted, unless the detector's own region is
        # confident (painted boundary lines) and the model is not: a weak automatic anchor never replaces it.
        if model.H is not None and model.level != LOW_LEVEL and (model.level == 'HIGH' or auto_conf < HIGH):
            outline = model.outline()
            if outline:
                mode = 'automatic' if model.level == 'HIGH' else 'robust'
                pitch = calibrated_model(pitch, outline, 'calibration' if model.source == 'calibration' else 'model', model.confidence,
                                         model.evidence, model.sides(), f'PITCH CONFIDENCE: {model.confidence:.2f} · {model.status_text()}', mode)
        if pitch.source in ('calibration', 'model'):
            self.mode = pitch.mode
            self._close_section(time)
            return pitch
        if auto_conf >= HIGH:
            pitch.mode, pitch.status = 'automatic', f'PITCH CONFIDENCE: {auto_conf:.2f} · PAINTED BOUNDARY LINES'
            self.mode = 'automatic'
            self._close_section(time)
            return pitch
        pitch.mode = 'robust'
        pitch.status = f'PITCH CONFIDENCE: {auto_conf:.2f} · GRASS REGION ONLY, NO PAINTED BOUNDARY (uncertain)'
        if model.lost:
            pitch.evidence = auto_evidence+[f'pitch model: {model.lost}']
        self.mode = 'robust'
        if tick and model.H is None and pitch.level == LOW_LEVEL:
            self._note_low(time, pitch)
            if self.low_since is not None and time-self.low_since >= ASSIST_AFTER:
                pitch.mode = 'assist'
                pitch.status += ' · NEEDS CONFIRMATION'
        return pitch

    def _note_low(self, time, pitch):
        if self.low_since is None:
            self.low_since, self.section = time, None
        # The representative frame: the halfway line in view, and as much grass as possible.
        score = (2.0 if halfway_candidate(pitch) is not None else 0.0)+pitch.coverage
        if self.section is None or score > self.section['score']:
            self.section = {'score': score, 'at': time, 'suggested': suggest_landmarks(pitch)}
        self.low_end = time

    def _close_section(self, time):
        if self.low_since is None:
            return
        start, end = self.low_since, getattr(self, 'low_end', self.low_since)
        if end-start >= ASSIST_AFTER and self.section is not None:
            self.sections.append({'start': start, 'end': end, 'at': self.section['at'], 'suggested': self.section['suggested']})
        self.low_since, self.section = None, None

    def pitch_issues(self):
        """LOW-confidence sections without a pitch model, nearby ones merged into one issue each."""
        self._close_section(self.last_time if self.last_time is not None else 0.0)
        merged = []
        for s in sorted(self.sections, key=lambda s: s['start']):
            if merged and s['start']-merged[-1]['end'] <= ASSIST_GAP:
                merged[-1]['end'] = max(merged[-1]['end'], s['end'])
                if len([m for m in s['suggested'] if m.get('predicted')]) > len([m for m in merged[-1]['suggested'] if m.get('predicted')]):
                    merged[-1]['at'], merged[-1]['suggested'] = s['at'], s['suggested']
            else:
                merged.append(dict(s))
        out = []
        for i, s in enumerate(merged):
            out.append({'id': f'pitch-{i+1}', 'start': round(s['start'], 2), 'end': round(s['end'], 2), 'at': round(s['at'], 2),
                        'suggested': s['suggested'],
                        'reason': f"Pitch confidence stayed low from {s['start']:.1f}s to {s['end']:.1f}s: no painted boundary line was found and no calibration "
                                  f"covers it, so the grass region was used. Mark a few landmarks on the frame at {s['at']:.1f}s to anchor the pitch model."})
        return out

    def snapshot(self):
        return {'registry': self.ids.snapshot(), 'teams': self.model.to_json()}
