"""Per-frame soccer tracking pipeline (everything after the detector):

    person detections -> camera motion -> pitch detection -> outside-field filter (foot point)
    -> local tracker (BoT-SORT) -> appearance (OSNet + part colours) -> kit votes + position cues
    -> pitch coordinates -> GLOBAL IDENTITY MANAGER (identity, team and role) -> persistent soccer
    observations; ball candidates -> one tracked match ball
"""
import math
from dataclasses import dataclass

import numpy as np

from . import appearance as app
from .ball import BallTracker
from .camera import CameraMotion, CameraState, Keyframes
from .geometry import iou
from .identity import IdentityManager, Options, Stab
from .pitch import FieldFilter, PitchDetector, calibrated_model, warp_model
from .relevance import REJECT_TEXT, assess
from .goals import GoalMemory, position_cues
from .teams import TeamModel, fit_team_model, kit_vote, with_keepers, with_referee

CLASSES = {1: 'goalkeeper', 2: 'player', 3: 'referee'}
MIN_VOTE_QUALITY = .2


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
        self.previous_boxes = []
        self.rejected_counts = {}
        self.events = []
        self.issues = []
        self.cuts = 0

    def _scale(self, boxes):
        """Expected person height (normalized) at a foot row: the perspective size model, else the median
        person on this frame."""
        if self.size is not None and self.size.reliable:
            return lambda y: max(1e-3, self.size.expected(y))
        heights = sorted(b[3] for b in boxes)
        return (lambda y: heights[len(heights)//2]) if heights else None

    def step(self, frame, time, rows, ball_rows=None):
        """rows: (N, 6) person detections [x1, y1, x2, y2, score, class id] in pixels; ball_rows: (M, 6)
        ball candidates from the detector."""
        W, H = self.width, self.height
        rows = np.asarray(rows, np.float32).reshape(-1, 6)
        dets = []
        for i, row in enumerate(rows):
            x1, y1, x2, y2 = np.clip(row[:4], [0, 0, 0, 0], [W, H, W, H])
            if x2-x1 < 2 or y2-y1 < 2:
                continue
            dets.append({'box': [float(x1/W), float(y1/H), float((x2-x1)/W), float((y2-y1)/H)], 'score': float(row[4]),
                         'cls': CLASSES.get(int(row[5]), 'player'), 'row': i})
        # 1. Camera motion: a pan is not player movement.
        M, reliable, cut = self.camera.estimate(frame, self.previous_boxes)
        ended = []
        if cut:
            self.cuts += 1
            ended += self.local.reset()
            self.pitch_detector.reset()
            self.size = None
        self.state.update(M, reliable, cut)
        self.keyframes.update(time, M, reliable)
        tick = cut or time-self.last_tick >= self.config.options.tick-1e-6
        if tick:
            self.last_tick = time
        # 2. Playable field, re-detected from the image on every evidence tick (never a fixed rectangle)
        # and carried with the camera motion in between.
        motion = self.keyframes.normalized_motion(M)
        if tick or self.analysed is None:
            self.analysed = self.pitch_detector.analyze(frame, time, motion @ self.carried_motion, reliable and self.carried_reliable, cut)
            self.carried_motion, self.carried_reliable = np.eye(3), True
            pitch = self.analysed
        else:
            self.carried_motion = motion @ self.carried_motion
            self.carried_reliable &= reliable
            pitch = warp_model(self.pitch, motion, time, reliable, cut)
        self.pitch = pitch
        outline = self.keyframes.pitch_outline()
        if outline:
            pitch = calibrated_model(pitch, outline)
        if self.size is not None and not cut:
            if reliable:
                Mn = self.keyframes.normalized_motion(M)
                self.size = self.size.moved(float(np.sqrt(abs(np.linalg.det(Mn[:2, :2])))), float(Mn[1, 2]))
            else:
                self.size = None
        # 3. Outside-field filter on the foot point.
        accepted, continuation, rejected, self.size = assess(dets, pitch, self.config.field_filter, self.size, self.keyframes.boundary())
        live = [tr.last_box for tr in self.ids.tracks.values() if tr.state in ('confirmed', 'uncertain') and time-tr.last_time < 1]
        continuation = [d for d in continuation if any(iou(d['box'], b) > .3 for b in live)]
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
            model = fit_team_model(self.ids.team_samples(current, time), self.model)
            self.model = with_keepers(with_referee(model, self.ids.referee_samples(model)), self.ids.keeper_samples())
            self.fit_at = time
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
            for sample in samples:
                sample['cues'] = cues[sample['id']]
        # 6. Global identity manager.
        result = self.ids.update({'time': time, 'tick': tick, 'samples': samples, 'ended': [{'id': i} for i in ended],
                                  'model': self.model, 'pitch_reliable': pitch.reliable or self.keyframes.boundary() is not None,
                                  'filter_enabled': self.config.field_filter.enabled, 'aspect': self.aspect,
                                  'segment': self.state.segment, 'drift': self.state.drift})
        for ghost in result['drop']:
            self.local.forget(ghost)
        # 7. The single match ball (its own tracker, separate from people).
        people = [(p['track'], p['box']) for p in result['people'] if p['state'] != 'rejected']
        people += [(None, b) for b in boxes if not any(iou(b, q) > .5 for _, q in people)]
        calibrated = self.keyframes.H is not None
        ball, ball_candidates = self.ball.step(frame, time, ball_rows if ball_rows is not None else np.zeros((0, 6)), M, reliable, cut,
                                               pitch, people, self._scale(boxes), self.state.stabilize_point, self.state.segment,
                                               self.keyframes.to_pitch_point if calibrated else None)
        self.events += result['events']+self.ball.events
        self.ball.events = []
        self.issues += result['issues']
        self.previous_boxes = boxes
        boundary = self.keyframes.boundary()
        return {'people': result['people'], 'ball': ball, 'ballCandidates': ball_candidates,
                'rejected': [{'box': [round(v, 4) for v in r['box']], 'reason': r['reason'], 'text': REJECT_TEXT[r['reason']],
                              **({'detail': r['detail'][:90]} if self.config.debug else {})} for r in rejected],
                'pitch': pitch.to_json() if (tick or cut) else None,
                'boundary': None if boundary is None else [[round(x, 4), round(y, 4)] for x, y in boundary],
                'cameraReliable': reliable, 'cut': cut, 'segment': self.state.segment, 'calibrated': self.keyframes.H is not None}, M, reliable

    def snapshot(self):
        return {'registry': self.ids.snapshot(), 'teams': self.model.to_json()}
