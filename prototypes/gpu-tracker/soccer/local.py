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


def _xywh(row):
    return [row[0], row[1], row[2]-row[0], row[3]-row[1]]


class BotSortTracker:
    def __init__(self, fps, use_embeddings=True):
        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.bot_sort import BOTSORT
        self._boxes = Boxes
        args = SimpleNamespace(track_high_thresh=HIGH, track_low_thresh=LOW, new_track_thresh=.40,
                               track_buffer=60, match_thresh=.75, fuse_score=True,
                               gmc_method='sparseOptFlow', proximity_thresh=.3, appearance_thresh=.85,
                               with_reid=False, model='auto')
        self.tracker = BOTSORT(args, frame_rate=max(1, round(fps)))
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
        subsets = {True: np.flatnonzero(conf >= HIGH), False: np.flatnonzero((conf > LOW) & (conf < HIGH))}
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
        alive = {t.track_id for t in self.tracker.tracked_stracks+self.tracker.lost_stracks}
        ended = [self.ids[(self.generation, t)] for t in self.known-alive if (self.generation, t) in self.ids]
        self.known = alive
        return matches, ended

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
