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
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from . import appearance as app
from .ball import BallTracker
from .camera import RELAY_BELOW, RELAY_OVERLAP, AnchorViews, CameraMotion, CameraState, Keyframes
from .geometry import foot_point, iou
from .identity import PLAYER_HEIGHT, IdentityManager, Options, Stab
from . import lens, midfield, visibility
from .local import HIGH as LOCAL_HIGH, NEW_TRACK as LOCAL_NEW_TRACK
from .pitch import (FieldFilter, PitchDetector, assess_confidence, calibrated_model, find_circle, halfway_candidate, line_samples, nearest_boundary,
                    suggest_landmarks, warp_model)
from .pitch_model import HIGH, LOW_LEVEL, MIN_ANCHOR
from .registration import Anchor
from .relevance import REJECT_TEXT, assess
from .goals import GoalMemory, position_cues
from .teams import TeamModel, fit_team_model, kit_vote, with_keepers, with_referee

CLASSES = {1: 'goalkeeper', 2: 'player', 3: 'referee'}
MIN_VOTE_QUALITY = .2
ASSIST_AFTER = 3.0        # seconds of LOW pitch confidence without a model before a section asks for confirmation
ASSIST_GAP = 3.0          # sections closer than this are one issue
ASSIST_PAN = 10.0         # ... and up to this far apart when one reliably measured camera movement joins them
FAR_MIN_PX = 30           # people expected smaller than this at the far edge of the pitch get the far-tile pass
FAR_CONFIRM = 3           # consecutive frames a far-tile detection must repeat before it counts
FAR_TRACK_SCORE = .45     # the tracking score a confirmed far-tile detection is given, so it can start a local track
SMALL_OBJECT_TRACKING = False  # BoT-SORT small-object tuning (local.small_objects) while the far band exists; kept only if measured to help
CROP_MAX_LOST = 1.0       # seconds: a confirmed player's lost local track is looked for with a crop this long
CROP_MAX_PX = 60          # ... when the person is this small or smaller (larger ones the whole-frame passes find)
CROP_SCALE = 3.0          # the crop is this many times the predicted box, at least CROP_MIN_PX wide
CROP_MIN_PX = 256
CROP_ACCEPT_IOU = .3      # a crop detection counts only this close to the prediction it was made for
CROP_ISOLATION_IOU = .1   # a crop detection overlapping another detected person more than this is not accepted (the lost player is behind them)
CROP_NEIGHBOUR = .5       # no crop for a lost player whose prediction, widened by this fraction of its size, touches another tracked person
CROPS_PER_FRAME = 4
REGISTER_EVERY = 1.0      # seconds between direct registrations of the pitch model to a remembered keyframe view
# The small-scene gate (SoccerTracker.small_scene): every amateur-only behaviour is behind it, and broadcast footage never
# opens it (its far band fired on 1 of 1800 frames of broadcast 0-60 s; amateur 3 at 118-182 s: 1191 of 1915 frames).
SMALL_SCENE_WINDOW = 2.0  # seconds of far-band history the gate looks at ...
SMALL_SCENE_SHARE = .5    # ... the far band active on at least this share of a full window of them opens it for the rest of the run
LOW_RES_HEIGHT = 720      # a source under this many rows is low resolution: always a small scene, and its pixel constants scale
REFERENCE_HEIGHT = 1080   # ... by height/REFERENCE_HEIGHT, the resolution they were tuned at (pixel_scale)
# Small scene only, each:
FAR_REPEAT_FOOT = .5      # a tiny far-tile row repeats (or continues a box) when its foot is within this many body heights ...
FAR_REPEAT_RATIO = 1.67   # ... and its height within this ratio either way: IoU > .5 never survives 1-2 px of jitter on a 6x13 px box
                          # (a replay of amateur 3's unconfirmed far rows: 6 of 1529 reached three repeats at IoU > .5, 131 under this rule)
BUFFER_MIN_H = 24         # px: BoT-SORT sees boxes shorter than this inflated to this height (buffered IoU) ...
BUFFER_MIN_W = 12         # px: ... and at least this wide, centred; the identity layer keeps the true box
DROPPED_PER_FRAME = 12    # debug: at most this many detections the local tracker dropped are listed per frame (the highest scores)


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


def pixel_scale(height):
    """Source pixels per pixel of the 1080p footage the pixel constants were tuned on (far band, crops, the small-person
    size tolerance), for a source under LOW_RES_HEIGHT rows: there every person is a few pixels tall while the detector
    input upscales the frame (an 896x506 source reaches the 1536 pass at 1.71x, a 1080p one at 0.8x), so absolute
    pixel thresholds would put everyone in the far band and run the crops at less magnification than the whole-frame
    pass. 1.0 for every source of LOW_RES_HEIGHT rows or more, so a 720p or 1080p run is unchanged."""
    return height/REFERENCE_HEIGHT if height < LOW_RES_HEIGHT else 1.0


def _buffered(xyxy):
    """A box shorter than BUFFER_MIN_H (or narrower than BUFFER_MIN_W) pixels inflated about its centre to that size, for
    BoT-SORT's IoU only: the detector's one or two pixels of jitter on a 6x13 px far player halve the IoU between two
    frames (an offset of 2 px gives .50), while the inflated pair keeps .71. Larger boxes pass unchanged, and the
    inflation is continuous in the box size, so nothing jumps where it starts."""
    x1, y1, x2, y2 = xyxy
    w, h = x2-x1, y2-y1
    if h >= BUFFER_MIN_H and w >= BUFFER_MIN_W:
        return xyxy
    cx, cy, w, h = (x1+x2)/2, (y1+y2)/2, max(w, BUFFER_MIN_W), max(h, BUFFER_MIN_H)
    return [cx-w/2, cy-h/2, cx+w/2, cy+h/2]


def _inside(box, other):
    """Share of a box's area inside another box: a small player behind a near one keeps a small IoU with them while
    most of the crop shows the other person."""
    ix = max(0.0, min(box[0]+box[2], other[0]+other[2])-max(box[0], other[0]))
    iy = max(0.0, min(box[1]+box[3], other[1]+other[3])-max(box[1], other[1]))
    return ix*iy/max(box[2]*box[3], 1e-9)


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
        self.views = AnchorViews(width, height)   # keyframe views for direct re-registration
        self.last_register = -math.inf
        self.registrations = {'tried': 0, 'adopted': 0, 'refused': 0, 'rootChanges': 0, 'relays': 0, 'viaRelay': 0}
        self.lens_samples = []  # per evidence tick: the lens estimate from the paint along the longest markings (soccer/lens.py)
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
        self.keyframe_log = []  # {'time', 'index', 'rms', 'observability'} for every calibration keyframe that anchored the model
        self.offset = 0         # frames of the log that precede the first emitted frame (the pre-roll)
        self.emitted = 0        # emitted frames (the segment itself); the log may continue past them (the post-roll)
        self.circle = None      # centre-circle paint found on the last evidence tick (pitch.find_circle), or None
        self.midfield = None    # the last automatic midfield registration attempt: {'time', 'taken', 'reason'|'observability'}
        self.registration = None   # filled by the pipeline's offline passes
        self.low_since = None   # start of the current LOW-confidence section without a model
        self.section = None     # its best representative tick so far
        self.sections = []      # closed LOW sections [{'start', 'end', 'at', 'suggested'}]
        self.far_pending = []   # far-tile detections waiting for confirmation: [box, hits, last time]
        self.mode = 'automatic'
        self.detection = Counter()   # how often each adaptive trigger fired (summary.detection)
        self.crop_predictions = []   # predicted boxes (normalized) the current frame's crop detections were made for
        self.fps = fps
        self.pixel_scale = pixel_scale(height)
        self.small_scene = False     # the small-scene gate, updated at the start of every step (_update_scene)
        self.far_history = deque()   # (time, far band active) over the last SMALL_SCENE_WINDOW seconds
        self.buffered_ids = set()    # local tracks BoT-SORT holds by a buffered box (_buffered); empty outside a small scene
        self.auto_landmarks = False  # set by pipeline.run: no calibration keyframe within the segment +- ROLL_SECONDS

    def _scale(self, boxes):
        """Expected person height (normalized) at a foot row: the perspective size model, else the median
        person on this frame."""
        if self.size is not None and self.size.reliable:
            return lambda y: max(1e-3, self.size.expected(y))
        heights = sorted(b[3] for b in boxes)
        return (lambda y: heights[len(heights)//2]) if heights else None

    def _live_boxes(self, time):
        # Ignored (non-target) people count too: their boxes are what tells a player "someone is overlapping you".
        return [tr.last_box for tr in self.ids.tracks.values() if tr.state in ('confirmed', 'uncertain', 'nontarget') and time-tr.last_time < 1]

    def _update_scene(self, time):
        """The small-scene gate for this frame: the far band (people expected under FAR_MIN_PX at the far edge of the
        pitch) active on at least SMALL_SCENE_SHARE of the frames of the last SMALL_SCENE_WINDOW seconds, counted over
        a full window so that the first second of a run cannot open it, or a low-resolution source. The band is
        decided from the previous frame's pitch and size model, exactly as the pipeline decided this frame's far tiles.

        Once open, the gate stays open for the rest of the sequence (time going backwards starts a new one). The far band
        comes and goes with the view while the footage stays what it is (the amateur 3 source: band on 62% of the frames,
        Veo 21%), and every change of the gate switches BoT-SORT's boxes of tiny people, the ball rules and the identity
        rules at once: a bare threshold would churn local tracks and identities wherever the band's share hovers around
        one half (a probe: four tiny people kept 4 local tracks with a steady gate, 10 to 17 with one flipping every 5
        frames). Broadcast never opens it, so nothing changes there."""
        restarted = bool(self.far_history) and time < self.far_history[-1][0]
        if restarted:
            self.far_history.clear()   # time went backwards: a new sequence
        self.far_history.append((time, self.far_band() is not None))
        while self.far_history[0][0] <= time-SMALL_SCENE_WINDOW+1e-9:
            self.far_history.popleft()
        active = sum(1 for _, on in self.far_history if on)
        opens = self.height < LOW_RES_HEIGHT or active >= SMALL_SCENE_SHARE*max(len(self.far_history), SMALL_SCENE_WINDOW*self.fps)
        return bool(opens or (self.small_scene and not restarted))

    def _scene_changed(self, opened):
        """The small-scene gate opened (or, on a new sequence, closed) on this frame. BoT-SORT associates tiny people by
        buffered boxes in a small scene only, so the live tracks' boxes are put into the new convention before this frame's
        update: otherwise a 6x13 px person's next detection overlaps their own track's 12x24 px box (or the reverse) at IoU
        .27, under what BoT-SORT matches, and every tiny person's local track restarts at the change (a probe with four
        tiny people: 8 local tracks instead of 4). Closing gives back the true size the identity layer kept; a track
        without one keeps its box."""
        W, H = self.width, self.height
        if not hasattr(self.local, 'rebox'):
            return
        if opened:
            def size(local_id, xyxy):
                box = _buffered(xyxy)
                return None if box is xyxy else box
            self.buffered_ids.update(i for i in self.local.rebox(size) if i is not None)
            return

        def size(local_id, xyxy):
            tr = self.ids.tracks.get(local_id) if local_id in self.buffered_ids else None
            if tr is None or tr.last_box is None:
                return None
            cx, cy, w, h = (xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2, tr.last_box[2]*W, tr.last_box[3]*H
            return [cx-w/2, cy-h/2, cx+w/2, cy+h/2]
        self.local.rebox(size)
        self.buffered_ids.clear()

    def _tiny(self, box):
        """A box the far-tile rules compare by foot point in a small scene: under FAR_MIN_PX source pixels tall."""
        return self.small_scene and box[3]*self.height < FAR_MIN_PX*self.pixel_scale

    def _same_spot(self, box, other):
        """Two boxes of one small person on nearby frames: feet within FAR_REPEAT_FOOT body heights, heights within
        FAR_REPEAT_RATIO of each other (in pixels, so the frame's aspect does not stretch the distance)."""
        ha, hb = box[3]*self.height, other[3]*self.height
        if min(ha, hb) <= 0 or max(ha, hb) > FAR_REPEAT_RATIO*min(ha, hb):
            return False
        dx = (box[0]+box[2]/2-other[0]-other[2]/2)*self.width
        dy = (box[1]+box[3]-other[1]-other[3])*self.height
        return math.hypot(dx, dy) <= FAR_REPEAT_FOOT*max(ha, hb)

    def _confirm_far(self, box, live, time):
        """A far-tile detection counts when it continues a tracked person or repeats on consecutive frames."""
        if self._tiny(box):
            return self._confirm_tiny(box, live, time)
        if any(iou(box, b) > .3 for b in live):
            return True
        self.far_pending = [p for p in self.far_pending if time-p[2] <= .5]
        for pending in self.far_pending:
            if iou(box, pending[0]) > .5:
                pending[0], pending[1], pending[2] = box, pending[1]+1, time
                return pending[1] >= FAR_CONFIRM
        self.far_pending.append([box, 1, time])
        return False

    def _confirm_tiny(self, box, live, time):
        """_confirm_far for a tiny box in a small scene: the same FAR_CONFIRM repeats within .5 s, and the same
        continuation of a tracked person, but a repeat or a continuation is the same spot (_same_spot) rather than an
        IoU. A pending detection counts at most once per frame, and the closest one takes the repeat."""
        if any(iou(box, b) > .3 or self._same_spot(box, b) for b in live):
            return True
        self.far_pending = [p for p in self.far_pending if time-p[2] <= .5]
        foot = lambda b: (b[0]+b[2]/2, b[1]+b[3])
        fx, fy = foot(box)
        near = [p for p in self.far_pending if p[2] < time and self._same_spot(box, p[0])]
        if near:
            pending = min(near, key=lambda p: math.hypot((foot(p[0])[0]-fx)*self.width, (foot(p[0])[1]-fy)*self.height))
            pending[0], pending[1], pending[2] = box, pending[1]+1, time
            return pending[1] >= FAR_CONFIRM
        self.far_pending.append([box, 1, time])
        return False

    def far_band(self):
        """Image rows (pixels) where people are expected to be too small for the whole-frame detector, for an
        extra tiled pass; None when nothing there is that small (the trigger is the expected size, whatever the
        pitch mode: broadcast footage, where the smallest players are about 40 px, never meets it). The pixel sizes
        scale with a low-resolution source (pixel_scale)."""
        pitch = self.pitch
        if pitch is None or not pitch.reliable or len(pitch.polygon) < 3 or self.size is None or not self.size.reliable:
            return None
        far_y = max(0.0, min(p[1] for p in pitch.polygon))
        least = FAR_MIN_PX*self.pixel_scale
        if self.size.expected(far_y+.02)*self.height >= least:
            return None
        y = far_y
        while y < far_y+.35 and self.size.expected(y)*self.height < 2*least:
            y += .01
        return int(max(0.0, far_y-.04)*self.height), int(min(1.0, y)*self.height)

    def lost_windows(self):
        """Crop windows (pixels) around the predicted boxes of confirmed players whose local track was lost less
        than CROP_MAX_LOST ago and who are small: where the detector looks again at full resolution. The
        predictions are kept so that step() accepts a crop detection only next to the one it was made for. The
        pixel sizes scale with a low-resolution source (pixel_scale), so its crops magnify like a 1080p frame's."""
        self.crop_predictions = []
        W, H = self.width, self.height
        windows = []
        for local_id, xyxy, gone in self.local.lost(int(round(CROP_MAX_LOST*self.fps))):
            tr = self.ids.tracks.get(local_id)
            if tr is None or tr.state != 'confirmed' or tr.last_box[3]*H > CROP_MAX_PX*self.pixel_scale:
                continue
            x1, y1, x2, y2 = xyxy
            cx, cy, w, h = (x1+x2)/2, (y1+y2)/2, max(1.0, x2-x1), max(1.0, y2-y1)
            if local_id in self.buffered_ids:
                # BoT-SORT followed this person with a buffered box (_buffered): its prediction has the inflated size
                # around the right centre, so the person's own size is put back (also once the gate has closed: the
                # crop's true-size detection would overlap the inflated prediction at IoU .27, under CROP_ACCEPT_IOU).
                w, h = max(1.0, tr.last_box[2]*W), max(1.0, tr.last_box[3]*H)
                x1, y1, x2, y2 = cx-w/2, cy-h/2, cx+w/2, cy+h/2
            if not (0 <= cx <= W and 0 <= cy <= H):
                continue
            # A player lost next to another tracked person is probably hidden behind them: a crop there would find
            # the other person and could carry this identity onto them, so the encounter is left to the identity
            # manager (which measures who comes out of it) and no crop is made.
            widened = [(x1-CROP_NEIGHBOUR*w)/W, (y1-CROP_NEIGHBOUR*h)/H, (w+2*CROP_NEIGHBOUR*w)/W, (h+2*CROP_NEIGHBOUR*h)/H]
            if any(other.last_box is not None and iou(widened, other.last_box) > 0 for lid, other in self.ids.tracks.items() if lid != local_id):
                self.detection['cropSkippedCrowded'] += 1
                continue
            half = max(CROP_MIN_PX*self.pixel_scale/2, CROP_SCALE*max(w, h)/2)
            window = (int(max(0, cx-half)), int(max(0, cy-half)), int(min(W, cx+half)), int(min(H, cy+half)))
            if window[2]-window[0] < 32 or window[3]-window[1] < 32:
                continue
            windows.append(window)
            self.crop_predictions.append([max(0.0, x1)/W, max(0.0, y1)/H, w/W, h/H])
            if len(windows) >= CROPS_PER_FRAME:
                break
        return windows

    def _dropped(self, local_rows, tracked):
        """The accepted detections the local tracker neither matched to a track nor started one with (local.unmatched):
        BoT-SORT drops them without a record, so a result could not tell 'not detected' from 'detected, then dropped by
        the tracker'. Output only (the frame's 'dropped' list, the highest scores first, and summary.detection counts):
        per entry the box, the score BoT-SORT saw and its bin (under .30 only a tracked track at IoU >= .5 takes it,
        .30-.40 cannot start a track, from .40 a miss is a mapping failure), the height in pixels, the best IoU with a
        track's box after the update and whether it came from the far tiles."""
        out = []
        for index, overlap in getattr(self.local, 'unmatched', ()):
            score, d = float(local_rows[index, 4]), tracked[index]
            band = '<.30' if score < LOCAL_HIGH else '.30-.40' if score < LOCAL_NEW_TRACK else '>=.40'
            self.detection[{'<.30': 'trackerDroppedLow', '.30-.40': 'trackerDroppedNoStart', '>=.40': 'trackerDroppedOther'}[band]] += 1
            out.append({'box': [round(v, 4) for v in d['box']], 'score': round(score, 3), 'bin': band, 'px': round(d['box'][3]*self.height, 1),
                        'trackIoU': round(float(overlap), 3), **({'far': True} if d.get('far') else {})})
        out.sort(key=lambda e: -e['score'])
        return out[:DROPPED_PER_FRAME]

    def step(self, frame, time, rows, ball_rows=None):
        """rows: (N, 6) person detections [x1, y1, x2, y2, score, class id] in pixels, or (N, 7) with a
        last column set for provisional far-tile detections (1) or track-guided crop detections (2);
        ball_rows: (M, 6) ball candidates, or (M, 7) with the candidate's source in the last column (0 the
        whole-frame passes, 1 the ball tiles, 2 the search crop), handed to the ball tracker as they are."""
        W, H = self.width, self.height
        rows = np.asarray(rows, np.float32)
        columns = 7 if rows.ndim == 2 and rows.shape[1] == 7 else 6
        rows = rows.reshape(-1, columns)
        kinds = rows[:, 6] if columns == 7 else np.zeros(len(rows), np.float32)
        rows = rows[:, :6]
        dt = max(0.0, time-self.last_time) if self.last_time is not None else 0.0
        self.last_time = time
        self.frame_index += 1
        scene = self._update_scene(time)
        if scene != self.small_scene:
            self._scene_changed(scene)
        self.small_scene = scene
        live = self._live_boxes(time)
        dets, unconfirmed = [], []
        # The whole-frame person detections (normalized boxes): a crop detection is accepted only away from them.
        whole = [[float(r[0]/W), float(r[1]/H), float((r[2]-r[0])/W), float((r[3]-r[1])/H)] for i, r in enumerate(rows) if kinds[i] != 2 and int(r[5]) != 0]
        # Small scene: the whole-frame passes' own people (box, score, row), which a tiny far-tile row of the same person
        # either duplicates or replaces (below).
        passes = [([float(r[0]/W), float(r[1]/H), float((r[2]-r[0])/W), float((r[3]-r[1])/H)], float(r[4]), i) for i, r in enumerate(rows)
                  if kinds[i] == 0 and int(r[5]) != 0] if self.small_scene else []
        superseded = set()   # whole-frame rows that a confirmed tiny far-tile row of the same person replaces
        for i, row in enumerate(rows):
            x1, y1, x2, y2 = np.clip(row[:4], [0, 0, 0, 0], [W, H, W, H])
            if x2-x1 < 2 or y2-y1 < 2:
                continue
            box = [float(x1/W), float(y1/H), float((x2-x1)/W), float((y2-y1)/H)]
            far = kinds[i] == 1
            # The whole-frame rows at the same spot as a tiny tile row: the same person, whose boxes' IoU is under the .3 of
            # pipeline.merge_far (a few pixels decide the IoU of a 13 px box).
            same = [(score, j) for o, score, j in passes if self._same_spot(box, o)] if far and self._tiny(box) else []
            if any(score >= LOCAL_NEW_TRACK for score, _ in same):
                # The whole-frame passes found this person scored to start or continue a local track by themselves: a
                # second detection of them would start a second local track once the tile row is confirmed.
                self.detection['farDuplicates'] += 1
                continue
            if far and not self._confirm_far(box, live, time):
                unconfirmed.append({'box': box, 'score': float(row[4]), 'reason': 'unconfirmed-far', 'detail': 'seen by the far-tile pass; not yet repeated or tracked'})
                continue
            if kinds[i] == 2:
                # A crop detection counts only next to the prediction the crop was made for.
                if not any(iou(box, p) >= CROP_ACCEPT_IOU for p in self.crop_predictions):
                    unconfirmed.append({'box': box, 'score': float(row[4]), 'reason': 'unconfirmed-crop', 'detail': 'found in a crop, away from the lost player it was looking for'})
                    continue
                if any(iou(box, o) > CROP_ISOLATION_IOU for o in whole):
                    # Next to another detected person: the lost player is probably behind them, and this box would
                    # carry their identity onto the other person.
                    self.detection['cropRejectedCrowded'] += 1
                    unconfirmed.append({'box': box, 'score': float(row[4]), 'reason': 'unconfirmed-crop', 'detail': 'found in a crop next to another person: the lost player is probably hidden behind them'})
                    continue
                self.detection['cropDetections'] += 1
            if far:
                self.detection['farDetections'] += 1
            if same:
                # Only weak whole-frame rows (under LOCAL_NEW_TRACK, so they start no local track) found this person, and
                # the far tiles a confirmed one: the tile row is the person's detection and the weak rows go, so that
                # BoT-SORT gets one detection of them and no second box of the same person marks them occluded.
                superseded.update(j for _, j in same)
            dets.append({'box': box, 'score': float(row[4]), 'cls': CLASSES.get(int(row[5]), 'player'), 'row': i, 'far': bool(far)})
        if superseded:
            self.detection['farSuperseded'] += sum(1 for d in dets if d['row'] in superseded)
            dets = [d for d in dets if d['row'] not in superseded]
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
        mask_boxes = [d['box'] for d in dets]
        if self.keyframes.anchored is not None and model.H is not None:
            # A validated landmark calibration: an anchor the offline passes carry both ways, and a view to
            # register later frames to directly.
            model.anchored_index = self.frame_index
            self.anchors.append(Anchor(self.frame_index, time, model.H.copy(), model.observability, 'landmarks', dict(model.details), root=self.frame_index))
            self.keyframe_log.append({'time': self.keyframes.anchored_time, 'index': self.frame_index, 'rms': self.keyframes.rms,
                                      'observability': round(float(model.observability), 3)})
            self.views.remember(self.frame_index, time, self.camera.previous, mask_boxes, model.H, model.observability, self.frame_index)
            self.last_register = time
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
            samples = line_samples(self.analysed.markings, self.pitch_detector.masks)
            if samples:
                self.lens_samples.append(lens.estimate_k([np.asarray(s)*[W, H] for s in samples], W, H))
            if model.H is not None:
                circle_px = [(x*W, y*H) for x, y in self.circle['points']] if self.circle else None
                model.refine(self.analysed.markings, time, [foot_point(b) for b in live], circle_px)
                if model.state == 'refined' and model.refinement is not None and model.refinement['own'] >= MIN_ANCHOR:
                    # A refinement anchors the offline registration only when its own paint constrains the pitch
                    # (MIN_ANCHOR); weaker matches still steady the live model but never restart a chain.
                    root = model.anchored_index if model.anchored_index is not None else self.frame_index
                    details = dict(model.details, score=model.refinement['own'], anchorObservability=round(model.refinement['anchor'], 3))
                    self.anchors.append(Anchor(self.frame_index, time, model.H.copy(), model.refinement['anchor'], 'paint', details, root=root))
            elif self.auto_landmarks or (self.circle is not None and auto_conf < HIGH):
                # No model to carry and no confident playable region from the paint: the centre circle, the
                # halfway line and a touchline can anchor one.
                self._midfield_anchor(model, circle_line, time, mask_boxes)
            if model.H is not None and self.views.views and time-self.last_register >= REGISTER_EVERY-1e-6:
                self._reregister(model, time, mask_boxes)
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
        # 3. Outside-field filter on the foot point (its pixel sizes in 1080p pixels: H itself unless low resolution).
        accepted, continuation, rejected, self.size = assess(dets, pitch, self.config.field_filter, self.size, self.keyframes.boundary(), live,
                                                             H/self.pixel_scale)
        continuation = [d for d in continuation if any(iou(d['box'], b) > .3 for b in live)]
        rejected += unconfirmed
        for r in rejected:
            self.rejected_counts[r['reason']] = self.rejected_counts.get(r['reason'], 0)+1
        tracked = accepted+continuation
        self.detection['farBoundaryContinued'] += sum(1 for d in tracked if d.get('continued') == 'far')
        # 4. Learned appearance for association and re-identification.
        xyxy = [[d['box'][0]*W, d['box'][1]*H, (d['box'][0]+d['box'][2])*W, (d['box'][1]+d['box'][3])*H] for d in tracked]
        embeddings = self.embedder(frame, xyxy) if self.embedder is not None and tracked else [None]*len(tracked)
        # A confirmed far-tile detection is given a tracking score that can start a local track (its own score,
        # from an upscaled tile, is below the tracker's new-track threshold); the person keeps the true score.
        def tracking_score(d):
            if d.get('far') and d['score'] < FAR_TRACK_SCORE:
                self.detection['farStarts'] += 1
                return FAR_TRACK_SCORE
            return d['score']
        # Small scene: BoT-SORT associates tiny people by buffered boxes (_buffered); the samples keep the true ones.
        association = xyxy
        if self.small_scene:
            association = [_buffered(b) for b in xyxy]
            self.detection['bufferedBoxes'] += sum(1 for a, b in zip(association, xyxy) if a is not b)
        local_rows = np.array([b+[tracking_score(d), int(rows[d['row'], 5])] for b, d in zip(association, tracked)], np.float32).reshape(-1, 6)
        feats = np.asarray(embeddings, np.float32) if self.embedder is not None and tracked else None
        small = bool(SMALL_OBJECT_TRACKING and self.far_band() is not None)
        if small:
            self.detection['smallTrackingFrames'] += 1
        if hasattr(self.local, 'small_objects'):
            self.local.small_objects(small)
        matches, gone = self.local.update(local_rows, frame, feats, M if reliable else None)
        ended += gone
        if self.small_scene or self.buffered_ids:
            # Which tracks BoT-SORT now holds by a buffered box: lost_windows gives their crop predictions the true size.
            for local_id, index in matches:
                if association[index] is xyxy[index]:
                    self.buffered_ids.discard(local_id)
                else:
                    self.buffered_ids.add(local_id)
            self.buffered_ids.difference_update(ended)
        dropped = self._dropped(local_rows, tracked)
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
        tracked_boxes = {local_id: tracked[index]['box'] for local_id, index in matches}
        for local_id, index in matches:
            d = tracked[index]
            pitch_point = self.keyframes.to_pitch(d['box'])
            sx, sy, sh = self.state.stabilize(d['box'])
            # How much of this box another person's box covers (the crop is then a mixture of two people), whose,
            # and how much of it lies inside another box: the identity manager freezes appearance, kit, role and
            # kinematic memory on such frames.
            overlap, overlap_with, covered = 0.0, None, 0.0
            for other_id, other_box in tracked_boxes.items():
                if other_id != local_id:
                    o = iou(d['box'], other_box)
                    if o > overlap:
                        overlap, overlap_with = o, other_id
                    covered = max(covered, _inside(d['box'], other_box))
            sample = {'id': local_id, 'box': d['box'], 'score': d['score'], 'cls': d['cls'], 'zone': d['zone'],
                      'occluded': _occluded(d['box'], boxes, self.aspect), 'stab': Stab(sx, sy, sh), 'overlap': round(overlap, 3), 'overlapWith': overlap_with,
                      'covered': round(covered, 3)}
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
                                  'small_scene': self.small_scene,
                                  'filter_enabled': self.config.field_filter.enabled, 'aspect': self.aspect,
                                  'segment': self.state.segment, 'drift': self.state.drift,
                                  'ball': self.last_ball if self.last_ball is not None and self.last_ball['segment'] == self.state.segment else None})
        for ghost in result['drop']:
            self.local.forget(ghost)
        # 7. The single match ball (its own tracker, separate from people).
        people = [(p['track'], p['box']) for p in result['people'] if p['state'] != 'rejected']
        people += [(None, b) for b in boxes if not any(iou(b, q) > .5 for _, q in people)]
        self.ball.small_scene = self.small_scene
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
                'pitch': self._tick_pitch(pitch, frame) if (tick or cut) else None,
                'boundary': None if boundary is None else [[round(x, 4), round(y, 4)] for x, y in boundary],
                'cameraReliable': reliable, 'cut': cut, 'segment': self.state.segment, 'calibrated': calibrated,
                'motion': [round(float(v), 5) for v in M[:2].ravel()] if reliable else None,
                'pitchCoordinates': ('calibration' if model.level == 'HIGH' else 'predicted') if calibrated else None,
                'pitchMode': pitch.mode, 'pitchConfidence': round(float(pitch.confidence), 3),
                'pitchModel': live_calibration if (tick or cut or self.config.debug) else None,
                # The frame's gates, for the offline passes (a run is a small scene when at least half its frames are).
                'profile': {'smallScene': self.small_scene},
                # Debug only: the detections the local tracker dropped, and the ball tracker's own diagnostics of the frame
                # (never in the frame's ball, which stays what the review compares).
                **({'dropped': dropped} if self.config.debug else {}),
                **({'ballDebug': self.ball.frame_debug} if self.config.debug and getattr(self.ball, 'frame_debug', None) is not None else {})}, M, reliable

    def _tick_pitch(self, pitch, frame):
        """The tick's pitch output; on small scenes only (never on broadcast), also the outline of the part of the frame that
        carries image (visibility.image_area), so the view does not count a dewarp's black as seen. No key when there is none."""
        out = pitch.to_json()
        area = visibility.image_area(frame) if self.small_scene else None
        if area is not None:
            out['imageArea'] = area
        return out

    def roll(self, frame, time):
        """A frame outside the segment, processed for its camera motion and pitch evidence only (no detections):
        a calibration keyframe shortly before the start is carried into the segment this way, and one shortly
        after the end is reached so that the offline registration can carry it backwards. The frame is
        logged like any other; nothing is emitted. Pitch events of such frames say so."""
        before = len(self.events)
        self.step(frame, time, np.zeros((0, 6), np.float32), np.zeros((0, 6), np.float32))
        for event in self.events[before:]:
            if event['kind'] == 'pitch':
                event['message'] += '\n(outside the segment: a keyframe carried in by camera motion)'

    def emitted_log(self):
        """The frame log of the emitted frames alone (the segment, without the pre-roll and post-roll)."""
        return self.frame_log[self.offset:self.offset+self.emitted]

    def _reregister(self, model, time, boxes):
        """Register this frame directly to a remembered keyframe view (camera.AnchorViews) and re-tie the model
        to that keyframe's geometry; adopted registrations are anchors of the offline registration, rooted at
        the keyframe. Events are logged when the view moved noticeably or the chain changed its root."""
        self.last_register = time
        self.registrations['tried'] += 1
        candidate = self.views.register(self.camera.previous, boxes, prefer=model.anchored_index)
        if candidate is None:
            return
        view = self.views.views[candidate['index']]
        H_direct = candidate['H'] @ view['H']
        if view['relay']:
            root_time = next((k['time'] for k in self.keyframe_log if k['index'] == view['root']), None)
            note = f"the relay view at {view['time']:.1f}s" + (f" (keyframe {root_time:.1f}s)" if root_time is not None else '')
        else:
            note = f"the keyframe view at {view['time']:.1f}s"
        shift = model.reregister(H_direct, time, candidate['matches'], candidate['rms'], note)
        if shift is None:
            self.registrations['refused'] += 1
            return
        self.registrations['adopted'] += 1
        self.registrations['viaRelay'] += bool(view['relay'])
        changed = model.anchored_index != view['index']
        if changed:
            self.registrations['rootChanges'] += 1
            model.anchored_index = view['index']
        quality = min(1.0, (candidate['matches']-40)/100)*max(0.0, 1-candidate['rms']/6.0)
        observability = view['observability']*(.8+.2*quality)
        details = {'matches': candidate['matches'], 'rmsPixels': round(candidate['rms'], 2), 'overlap': round(candidate['overlap'], 3),
                   'shiftPixels': round(shift, 1), 'score': round(observability, 3), 'view': view['index'], 'relay': bool(view['relay'])}
        self.anchors.append(Anchor(self.frame_index, time, model.H.copy(), observability, 'reregistered', details, root=view['root']))
        if candidate['matches'] < RELAY_BELOW or candidate['overlap'] < RELAY_OVERLAP:
            # The view is leaving the frame: keep this re-registered frame as a relay, so the next frames register to
            # it (rooted at the same keyframe) instead of falling back to the frame-to-frame chain.
            if self.views.remember(self.frame_index, time, self.camera.previous, boxes, model.H, observability, view['root'], relay=True):
                self.registrations['relays'] += 1
                model.anchored_index = self.frame_index

    def _midfield_anchor(self, model, halfway, time, boxes=()):
        """Automatic registration from the centre circle (an ellipse), the halfway line and a touchline
        (soccer/midfield.py), validated against the paint and scored for observability before it anchors
        the pitch model; a refused attempt is reported once per reason.

        A run without user keyframes near it (auto_landmarks, set by the pipeline) searches the circle on the
        grass with the people masked (pitch.find_circle on_grass: the shipped search fits the background on a
        far, flat circle) and solves a pinhole camera (midfield.solve allow_pinhole, with the people's heights
        and the frame's other markings); its observability ceiling (cap) goes with the anchor, and so do the
        painted lines it was fitted to (fitted), which can never lift that ceiling. Runs with keyframes keep the
        shipped search and solver."""
        W, H = self.width, self.height
        auto = bool(getattr(self, 'auto_landmarks', False))
        circle = self.circle
        if auto:
            circle, halfway = find_circle(self.pitch_detector.masks, self.analysed, on_grass=True, boxes=boxes)
            if circle is None:
                reason = 'no centre circle on the grass'
                if self.midfield is None or self.midfield.get('reason') != reason:
                    model.events.append((time, 'midfield', f'no automatic anchor: {reason}'))
                self.midfield = {'time': time, 'taken': False, 'reason': reason, 'route': 'pinhole'}
                return
        ellipse = midfield.ellipse_from_points([(x*W, y*H) for x, y in circle['points']])
        if ellipse is None:
            return
        touchlines = midfield.touchline_candidates(self.analysed, ellipse, W, H)
        if auto:
            result = midfield.solve(circle, halfway, touchlines, W, H, self.keyframes.size, self.pitch_detector.masks, allow_pinhole=True,
                                    people=list(boxes), markings=self.analysed.markings)
        else:
            result = midfield.solve(circle, halfway, touchlines, W, H, self.keyframes.size, self.pitch_detector.masks)
        if 'H' not in result:
            if self.midfield is None or self.midfield.get('reason') != result['reason']:
                model.events.append((time, 'midfield', f'centre circle seen ({circle["coverage"]:.0%} of the arc) but not used as an anchor: {result["reason"]}'))
            self.midfield = {'time': time, 'taken': False, 'reason': result['reason']} | ({'route': 'pinhole'} if auto else {})
            return
        if auto:
            d = result['details']
            note = ', '.join(d.get('constraints') or d.get('evidence') or ['centre circle', 'halfway line'])+(' (pinhole camera)' if d.get('route') == 'pinhole' else ' (assumed width)')
            taken = model.anchor(np.linalg.inv(result['H']), time, note=note, observability=result['observability'], details=result['details'],
                                 source='auto-midfield', cap=result.get('cap'), fitted=result.get('fitted'))
        else:
            taken = model.anchor(np.linalg.inv(result['H']), time, note='centre circle, halfway line and touchline', observability=result['observability'],
                                 details=result['details'], source='auto-midfield')
        self.midfield = {'time': time, 'taken': bool(taken), 'observability': round(result['observability'], 3), 'rmsPixels': round(result['rms_px'], 1),
                         'evidence': result['details'].get('evidence')} | ({'route': result['details'].get('route')} if auto else {})
        if taken:
            model.anchored_index = self.frame_index
            self.anchors.append(Anchor(self.frame_index, time, model.H.copy(), model.observability, 'auto-midfield', dict(model.details), root=self.frame_index))
            self.views.remember(self.frame_index, time, self.camera.previous, list(boxes), model.H, model.observability, self.frame_index)
            self.last_register = time

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

    def _continuous(self, t0, t1):
        """Was the camera motion measured reliably, without a cut, all the way from t0 to t1? Then two uncovered
        sections belong to one continuous pan and are asked about once."""
        between = [e for e in self.frame_log if t0-1e-6 <= e['t'] <= t1+1e-6]
        return bool(between) and all(e['reliable'] and not e['cut'] for e in between)

    def _landmark_hint(self, at):
        """What the nearest anchor's evidence lacked, as the landmark family worth marking."""
        if not self.anchors:
            return 'four or more landmarks spanning both directions of the pitch (a touchline end, the centre spot, a box corner)'
        nearest = min(self.anchors, key=lambda a: abs(a.time-at))
        d = nearest.details or {}
        if d.get('landmarks', 0) >= 4:
            return 'landmarks on both halves of the pitch, the near touchline included when it is visible (the earlier keyframe lies on one side only)' \
                if d.get('spread', 1.0) < .5 else 'a few landmarks on this frame; the earlier keyframe is too far away to carry here'
        if not d.get('transverse'):
            return 'a goal line, a box front or the halfway line: the touchlines alone leave the length of the pitch unresolved'
        if not d.get('longitudinal'):
            return 'a touchline or a box side: the lines across the pitch alone leave its width unresolved'
        return 'a few landmarks on this frame'

    def pitch_issues(self):
        """LOW-confidence sections without a pitch model, merged into one issue when they are close or when one
        continuous, reliably measured camera movement joins them (one pan asks once); each names the landmark
        family the nearby evidence lacked."""
        self._close_section(self.last_time if self.last_time is not None else 0.0)
        merged = []
        for s in sorted(self.sections, key=lambda s: s['start']):
            gap = s['start']-merged[-1]['end'] if merged else math.inf
            if merged and (gap <= ASSIST_GAP or (gap <= ASSIST_PAN and self._continuous(merged[-1]['end'], s['start']))):
                merged[-1]['end'] = max(merged[-1]['end'], s['end'])
                if len([m for m in s['suggested'] if m.get('predicted')]) > len([m for m in merged[-1]['suggested'] if m.get('predicted')]):
                    merged[-1]['at'], merged[-1]['suggested'] = s['at'], s['suggested']
            else:
                merged.append(dict(s))
        out = []
        for i, s in enumerate(merged):
            hint = self._landmark_hint(s['at'])
            out.append({'id': f'pitch-{i+1}', 'start': round(s['start'], 2), 'end': round(s['end'], 2), 'at': round(s['at'], 2),
                        'suggested': s['suggested'], 'hint': hint,
                        'reason': f"Pitch confidence stayed low from {s['start']:.1f}s to {s['end']:.1f}s: no painted boundary line was found and no calibration "
                                  f"covers it, so the grass region was used. Mark landmarks on the frame at {s['at']:.1f}s to anchor the pitch model: {hint}."})
        return out

    def snapshot(self):
        return {'registry': self.ids.snapshot(), 'teams': self.model.to_json()}
