"""Local multi-object tracker adapter around Ultralytics BoT-SORT.

BoT-SORT follows visible objects; it does not know who they are. Its IDs restart after a reset and
are never reused here: every (generation, BoT-SORT id) pair gets a fresh local track number. The
identity manager decides who a local track is. Ultralytics is imported lazily so the rest of the
soccer package stays importable without torch.
"""
from types import SimpleNamespace

import numpy as np

from .geometry import iou

HIGH, LOW = .30, .10
NEW_TRACK = .40          # a detection needs this score to start a track (an unmatched one scoring HIGH..NEW_TRACK is dropped)
SMALL_LOW = .05          # second-stage threshold while small people are expected: weaker detections still continue a track
SMALL_BUFFER = 90        # frames an unmatched track survives while small people are expected (60 otherwise)
BUFFER = 60


def _xywh(row):
    return [row[0], row[1], row[2]-row[0], row[3]-row[1]]


class BotSortTracker:
    def __init__(self, fps, use_embeddings=True):
        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.bot_sort import BOTSORT
        self._boxes = Boxes
        args = SimpleNamespace(track_high_thresh=HIGH, track_low_thresh=LOW, new_track_thresh=NEW_TRACK,
                               track_buffer=60, match_thresh=.75, fuse_score=True,
                               gmc_method='sparseOptFlow', proximity_thresh=.3, appearance_thresh=.85,
                               with_reid=False, model='auto')
        self.tracker = BOTSORT(args, frame_rate=max(1, round(fps)))
        self.fps = max(1, round(fps))
        self.low = LOW
        self.small = False
        self.use_embeddings = use_embeddings
        # Features are computed once per frame by the pipeline (OSNet) and passed in.
        self.tracker.encoder = lambda feats, boxes: [np.asarray(f, np.float32) for f in feats]
        # Camera motion is estimated once per frame by camera.py and shared with BoT-SORT's Kalman
        # prediction (instead of BoT-SORT computing its own optical flow a second time).
        self.motion = np.eye(2, 3)
        self.tracker.gmc.apply = lambda img, dets=None: self.motion
        self.generation = 0
        self.ids = {}
        self.next = 1
        self.known = set()
        self.unmatched = []   # after each update: [(row index, best IoU with a track)] for rows neither matched nor starting a track

    def _local(self, track_id):
        key = (self.generation, int(track_id))
        if key not in self.ids:
            self.ids[key] = self.next
            self.next += 1
        return self.ids[key]

    def update(self, rows, frame, feats=None, motion=None):
        """rows: (N, 6) [x1, y1, x2, y2, score, class] in pixels; motion: 3x3 camera transform from the
        previous frame (identity when unknown). Returns ([(local id, row index)], ended ids)."""
        rows = np.asarray(rows, np.float32).reshape(-1, 6)
        self.motion = np.eye(2, 3) if motion is None else np.asarray(motion, np.float64)[:2].copy()
        boxes = self._boxes(rows, frame.shape[:2])
        # Appearance features are used only when supplied for every detection.
        use = self.use_embeddings and feats is not None and len(feats) == len(rows)
        self.tracker.args.with_reid = use
        output = self.tracker.update(boxes, frame, np.asarray(feats, np.float32)) if use else self.tracker.update(boxes, frame)
        conf = rows[:, 4]
        subsets = {True: np.flatnonzero(conf >= HIGH), False: np.flatnonzero((conf > self.low) & (conf < HIGH))}
        matches, used = [], set()
        for out in output:
            box, track_id, score, idx = out[:4], int(out[4]), float(out[5]), int(out[7])
            pool = subsets[score >= HIGH]
            det = int(pool[idx]) if 0 <= idx < len(pool) else -1
            if det < 0 or det in used or iou(_xywh(rows[det]), _xywh(box)) < .3:
                best = max((i for i in range(len(rows)) if i not in used), key=lambda i: iou(_xywh(rows[i]), _xywh(box)), default=-1)
                det = best if best >= 0 and iou(_xywh(rows[best]), _xywh(box)) >= .3 else -1
            if det >= 0:
                used.add(det)
                matches.append((self._local(track_id), det))
        self.unmatched = self._unmatched(rows, used, subsets[True])
        alive = {t.track_id for t in self.tracker.tracked_stracks+self.tracker.lost_stracks}
        ended = [self.ids[(self.generation, t)] for t in self.known-alive if (self.generation, t) in self.ids]
        self.known = alive
        return matches, ended

    def _unmatched(self, rows, used, high):
        """Rows the update neither matched to an output track nor started a new (not yet confirmed) track with, each with
        its best IoU against the tracks' boxes after the update (a lost track's is its prediction). A track started on
        this frame is recognised by its start frame, and its detection by its index in the high-score pool. Read only."""
        now = self.tracker.frame_id
        started = {int(high[int(t.idx)]) for t in self.tracker.tracked_stracks
                   if not t.is_activated and t.start_frame == now and 0 <= int(t.idx) < len(high)}
        left = [i for i in range(len(rows)) if i not in used and i not in started]
        if not left:
            return []
        tracks = np.asarray([t.xyxy for t in self.tracker.tracked_stracks+self.tracker.lost_stracks], np.float64).reshape(-1, 4)
        out = []
        for i in left:
            r = rows[i]
            if len(tracks):
                ix = np.maximum(0, np.minimum(r[2], tracks[:, 2])-np.maximum(r[0], tracks[:, 0]))
                iy = np.maximum(0, np.minimum(r[3], tracks[:, 3])-np.maximum(r[1], tracks[:, 1]))
                inter = ix*iy
                union = (r[2]-r[0])*(r[3]-r[1])+(tracks[:, 2]-tracks[:, 0])*(tracks[:, 3]-tracks[:, 1])-inter
                best = float((inter/np.maximum(union, 1e-9)).max())
            else:
                best = 0.0
            out.append((i, best))
        return out

    def small_objects(self, active):
        """Small-object tuning while the measured condition holds (people expected under the far-tile size): a lower
        second-stage threshold and a longer buffer; the defaults otherwise. Takes effect from the next update."""
        active = bool(active)
        if active == self.small:
            return
        self.small = active
        self.low = SMALL_LOW if active else LOW
        self.tracker.args.track_low_thresh = self.low
        self.tracker.max_time_lost = int(self.fps/30.0*(SMALL_BUFFER if active else BUFFER))

    def lost(self, max_frames):
        """Local tracks lost for at most max_frames frames: [(local id, predicted [x1, y1, x2, y2] pixels, frames lost)]."""
        out = []
        now = self.tracker.frame_id
        for track in self.tracker.lost_stracks:
            gone = now-track.end_frame
            if 0 < gone <= max_frames and (self.generation, track.track_id) in self.ids:
                out.append((self.ids[(self.generation, track.track_id)], [float(v) for v in track.xyxy], int(gone)))
        return out

    def rebox(self, size):
        """Give live tracks (tracked or lost) a new box: size(local id or None, [x1, y1, x2, y2] pixels) returns the new
        box, or None to keep the track's. The Kalman state's position and size are replaced, its velocities and
        covariance kept, so the next prediction starts from the new box. For a change of the convention the boxes are
        given in (soccer.tracker: buffered boxes for tiny people in a small scene). Returns the local ids changed."""
        changed = []
        for track in self.tracker.tracked_stracks+self.tracker.lost_stracks:
            if track.mean is None:
                continue
            local_id = self.ids.get((self.generation, int(track.track_id)))
            box = size(local_id, [float(v) for v in track.xyxy])
            if box is None:
                continue
            x1, y1, x2, y2 = box
            track.mean[:4] = track.convert_coords(np.asarray([x1, y1, x2-x1, y2-y1], np.float64))
            changed.append(local_id)
        return changed

    def forget(self, local_id):
        """Remove a lost track whose identity moved to another track, so it cannot come back as a ghost."""
        ids = [tid for (gen, tid), lid in self.ids.items() if lid == local_id and gen == self.generation]
        if not ids:
            return
        tid = ids[0]
        for pool in (self.tracker.tracked_stracks, self.tracker.lost_stracks):
            for track in list(pool):
                if track.track_id == tid:
                    pool.remove(track)
                    track.mark_removed()
                    self.tracker.removed_stracks.append(track)
        self.known.discard(tid)

    def reset(self):
        """Camera cut: every local track ends. Returns their ids."""
        ended = [self.ids[(self.generation, t)] for t in self.known if (self.generation, t) in self.ids]
        self.tracker.reset()
        self.generation += 1
        self.known = set()
        return ended
