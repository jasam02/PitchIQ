"""Local CUDA prototype. No uploads, API calls, roster guesses or off-screen paths.

Detection (soccer YOLOv8x on CUDA) feeds the soccer tracking layer in soccer/: pitch detection,
foot-point field filtering, BoT-SORT local tracks, OSNet + kit appearance, team/role votes and the
global identity manager that keeps one identity per real person across exits and returns.
"""
import os
import json
import queue
import threading
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
(ROOT / 'settings' / 'Ultralytics').mkdir(parents=True, exist_ok=True)
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / 'settings')
os.environ['YOLO_AUTOINSTALL'] = 'false'

import cv2
import numpy as np
import torch
from ultralytics import YOLO, settings
from torchvision.ops import nms
from download_model import get_model
from soccer import ball_path, imputation, labels, lens, pitch_model, registration, roster, tracklets, visibility
from soccer.identity import long_id
from soccer.local import BotSortTracker
from soccer.tracker import SoccerTracker, config_from
from vision import Appearance

settings.update({'sync': False})
ROLL_SECONDS = 15.0   # a calibration keyframe this close before the start (or after the end) is carried into the segment by camera motion
DETECT_BATCH = 1      # consecutive frames per whole-frame detector call on the prefetch worker (config detectBatch). One keeps the
                      # detections bit for bit those of a serial call; on the RTX 5080 larger batches bought no measurable time
                      # because the main thread is the bottleneck. Worth trying again on a different GPU.
BALL_SCAN_S = 1.0     # seconds between the safeguard ball tile scans while the ball tracker is LOCKED (config ballScanEvery);
                      # while it is not (searching, recovering, occluded) the tiles run every other frame
NEAR_PASS = 640       # small scenes only: an extra whole-frame people pass at this input size (config nearPass). On amateur 3 (1080p,
                      # pole camera) the 960 and 1536 passes miss near players of 60-120 px that a 640 pass finds (148.0 s: the ball
                      # carrier is found at 640 only; 1280 and 1536 often find no near player at all)
NEAR_MIN_PX = 60      # px at 1080p (scaled with a low-resolution source): the near pass only fills in people at least this tall ...
NEAR_FILL_SHARE = .3  # ... that no whole-frame person row covers this share of (of the smaller box). Merged in the joint
                      # suppression it replaced the main passes' box on about 3 people per frame (24-60 px) and put a second box on
                      # (or one over two of) about 2 more, which cut their local tracks (amateur 3 118-182 s: 357 -> 524 local
                      # tracks; 38 sampled frames: 6 rows of 60 px or more that the other passes miss, 2 under it)
BALL_TILE_PX = 800    # px: the ball tile side, shrunk to 640 (0.8x, at 1080p; scaled with a low-resolution source)
# The ball candidates' source, the last column of the rows handed to the tracker (diagnostic: no decision reads it).
BALL_FROM_PASSES, BALL_FROM_TILES, BALL_FROM_CROP = 0.0, 1.0, 2.0
# BGR; the same colours as the review page.
COLORS = {'A': (255, 140, 185), 'B': (100, 181, 246), 'GOALKEEPER': (162, 95, 255), 'REFEREE': (100, 225, 255), 'UNCERTAIN': (194, 180, 169),
          'PITCH': (255, 200, 124), 'BALL': (61, 255, 198)}


def metadata(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError('This video could not be opened.')
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or count <= 0:
            raise ValueError('Could not read video timing.')
        return {'name': Path(path).name, 'width': int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 'fps': fps,
                'frames': count, 'duration': count/fps}
    finally:
        capture.release()


def gpu_info():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available. Run setup.cmd and check your NVIDIA driver. CPU fallback is intentionally disabled.')
    # Execute a real CUDA kernel, rather than relying on device enumeration.
    value = torch.ones((16, 16), device='cuda') @ torch.ones((16, 16), device='cuda')
    torch.cuda.synchronize()
    if value[0, 0].item() != 16:
        raise RuntimeError('CUDA could not execute correctly.')
    return {'name': torch.cuda.get_device_name(0), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'memoryGB': round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1)}


def normalized(box, width, height):
    x1, y1, x2, y2 = np.clip(box, [0, 0, 0, 0], [width, height, width, height])
    return [round(float(x1/width), 6), round(float(y1/height), 6),
            round(float((x2-x1)/width), 6), round(float((y2-y1)/height), 6)]


def ball_tiles(model, frame, scale=1.0):
    """The whole image in overlapping square regions of BALL_TILE_PX, each run at 640 (0.8x on a 1080p frame: the
    scale of a 1536 detail pass, more than a 960 or 1280 one), so the ball is looked for everywhere, not only where
    it is expected. scale: tracker.pixel_scale, under 1 for a low-resolution source, whose regions shrink with it
    so that the ball reaches the detector at the size it has in a 1080p frame's tiles."""
    height, width = frame.shape[:2]
    size = min(round(BALL_TILE_PX*scale), width, height)
    def starts(length):
        return sorted(set(list(range(0, max(1, length-size), round(size*.8)))+[length-size]))
    regions = [(x, y) for y in starts(height) for x in starts(width)]
    output = []
    # All tiles in one call (six 640 px tiles of a 1080p frame: a few hundred MB at most).
    predictions = model.predict([frame[y:y+size, x:x+size] for x, y in regions],
                                imgsz=640, conf=.09, iou=.4, classes=[0], device=0, half=True, verbose=False)
    for (x, y), prediction in zip(regions, predictions):
        rows = prediction.boxes.data.cpu().numpy().copy()
        if len(rows):
            rows[:, :4] += [x, y, x, y]
            output.extend(rows.tolist())
    return output


def ball_crop(model, frame, window, scale=1.0):
    """Ball detections in a crop of the frame at its original resolution (the local search around the
    ball tracker's prediction): rows [x1, y1, x2, y2, score, 0] in frame pixels. scale: tracker.pixel_scale; a
    low-resolution source's crop is magnified by 1/scale (up to 1280), as a 1080p frame's crop would show the ball."""
    x0, y0, x1, y1 = window
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    size = int(min(1280, max(320, 32*round(max(crop.shape[:2])/scale/32))))
    result = model.predict(crop, imgsz=size, conf=.08, iou=.4, classes=[0], device=0, half=True, verbose=False)[0]
    rows = []
    for box in result.boxes:
        bx1, by1, bx2, by2 = box.xyxy[0].tolist()
        rows.append([bx1+x0, by1+y0, bx2+x0, by2+y0, float(box.conf[0]), 0.0])
    return rows


def far_tiles(model, frame, band):
    """Robust-footage fallback: overlapping square tiles over the image band (y0, y1) in pixels where
    the tracked people are expected to be too small for the whole-frame passes, run at a higher
    effective resolution. Only people are looked for; the tracker treats the result as provisional
    (pipeline column 7) until it repeats or continues a tracked person. The tiles stay at least 96 source
    pixels wide on any source (the band itself scales with a low-resolution one): that caps the magnification
    of source pixels at 6.7x, and a low-resolution source has no finer detail to magnify."""
    y0, y1 = band
    height, width = frame.shape[:2]
    size = max(96, y1-y0)
    y0 = max(0, min(y0, height-size))
    starts = sorted(set(list(range(0, max(1, width-size), round(size*.75)))+[max(0, width-size)]))
    out = []
    predictions = model.predict([frame[y0:y0+size, x:x+size] for x in starts], imgsz=640, conf=.15, iou=.45,
                                classes=[1, 2, 3], device=0, half=True, verbose=False)
    for x, prediction in zip(starts, predictions):
        rows = prediction.boxes.data.cpu().numpy().copy()
        if len(rows):
            rows[:, :4] += [x, y0, x, y0]
            out.extend(rows.tolist())
    rows = np.asarray(out, np.float32).reshape(-1, 6)
    if len(rows) > 1:
        # The tiles overlap by a quarter: one person on the seam is found twice; keep the better box.
        raw = torch.tensor(rows)
        rows = rows[nms(raw[:, :4], raw[:, 4], .5).numpy()]
    return rows


def local_crops(model, frame, windows):
    """Track-guided re-detection: the detector on crops around the predicted boxes of confirmed players whose
    local track was just lost and who are small (pipeline column 7 = 2; the tracker accepts such a row only
    next to its prediction). windows: [(x0, y0, x1, y1)] in pixels, at most a few per frame."""
    crops, offsets = [], []
    for x0, y0, x1, y1 in windows:
        crop = frame[y0:y1, x0:x1]
        if crop.size and min(crop.shape[:2]) >= 32:
            crops.append(crop)
            offsets.append((x0, y0))
    if not crops:
        return np.zeros((0, 7), np.float32)
    size = int(min(640, max(256, 32*round(max(max(c.shape[:2]) for c in crops)/32))))
    predictions = model.predict(crops, imgsz=size, conf=.10, iou=.45, classes=[1, 2, 3], device=0, half=True, verbose=False)
    out = []
    for (x0, y0), prediction in zip(offsets, predictions):
        rows = prediction.boxes.data.cpu().numpy().copy()
        if len(rows):
            rows[:, :4] += [x0, y0, x0, y0]
            out.extend(np.c_[rows, np.full(len(rows), 2.0)].tolist())
    return np.asarray(out, np.float32).reshape(-1, 7)


def merge_crops(rows, extra):
    """Rows (N, 6 or 7) plus crop rows (M, 7): a crop row that the whole-frame passes already cover (IoU >= .5
    with any person row) is dropped; the rest are appended with their flag."""
    if rows.shape[1] == 6:
        rows = np.c_[rows, np.zeros(len(rows), np.float32)]
    if not len(extra):
        return rows
    people = rows[rows[:, 5] != 0]
    keep = []
    for r in extra:
        if len(people):
            ix = np.maximum(0, np.minimum(r[2], people[:, 2])-np.maximum(r[0], people[:, 0]))
            iy = np.maximum(0, np.minimum(r[3], people[:, 3])-np.maximum(r[1], people[:, 1]))
            inter = ix*iy
            union = (r[2]-r[0])*(r[3]-r[1])+(people[:, 2]-people[:, 0])*(people[:, 3]-people[:, 1])-inter
            if (inter/np.maximum(union, 1e-6)).max() >= .5:
                continue
        keep.append(r)
    return np.r_[rows, np.asarray(keep, np.float32).reshape(-1, 7)]


def merge_far(rows, extra):
    """Whole-frame person rows (N, 6) plus far-tile rows (M, 6) -> (K, 7): the last column marks the
    tile detections that no whole-frame detection already covers."""
    if not len(extra):
        return np.c_[rows, np.zeros(len(rows), np.float32)]
    people = rows[rows[:, 5] != 0]
    keep = []
    for r in extra:
        ix = np.maximum(0, np.minimum(r[2], people[:, 2])-np.maximum(r[0], people[:, 0])) if len(people) else np.zeros(0)
        iy = np.maximum(0, np.minimum(r[3], people[:, 3])-np.maximum(r[1], people[:, 1])) if len(people) else np.zeros(0)
        inter = ix*iy
        union = (r[2]-r[0])*(r[3]-r[1])+(people[:, 2]-people[:, 0])*(people[:, 3]-people[:, 1])-inter if len(people) else np.zeros(0)
        if not len(people) or (inter/np.maximum(union, 1e-6)).max() < .3:
            keep.append(r)
    flagged = np.c_[np.asarray(keep, np.float32).reshape(-1, 6), np.ones(len(keep), np.float32)]
    return np.r_[np.c_[rows, np.zeros(len(rows), np.float32)], flagged]


def ball_candidates(rows, tile_rows, crop_rows, width, height):
    """One frame's ball candidates from its three sources: the whole-frame passes' ball rows (rows: their (N, 6)
    detections), the ball tiles' rows (None on the frames the scans skip) and the search crop's. One suppression
    at IoU .3 merges them (the better-scored box stays, whatever its source) and obviously huge boxes (largest
    side over 3.5% of the frame) go; everything else is scored by the ball tracker. Returns the (M, 7) rows
    [x1, y1, x2, y2, score, 0, source] with the source (BALL_FROM_*) carried through, and the diagnostic counts per
    source before and after: {'raw': [passes, tiles, crop], 'kept': [...]}."""
    found = [r+[BALL_FROM_PASSES] for r in rows[rows[:, 5] == 0].tolist()]
    found += [list(r)+[BALL_FROM_TILES] for r in (tile_rows or [])]
    found += [list(r)+[BALL_FROM_CROP] for r in crop_rows]
    counts = lambda sources: [int(np.sum(sources == s)) for s in (BALL_FROM_PASSES, BALL_FROM_TILES, BALL_FROM_CROP)]
    if not found:
        return np.zeros((0, 7), np.float32), {'raw': [0, 0, 0], 'kept': [0, 0, 0]}
    raw = torch.tensor(found, dtype=torch.float32)
    kept = raw[nms(raw[:, :4], raw[:, 4], .3)].numpy()
    kept = kept[np.maximum(kept[:, 2]-kept[:, 0], kept[:, 3]-kept[:, 1]) <= max(width, height)*.035]
    return np.asarray(kept, np.float32).reshape(-1, 7), {'raw': counts(raw[:, 6].numpy()), 'kept': counts(kept[:, 6])}


def auto_landmarks(keyframe_times, start, finish):
    """Whether this run must find its pitch landmarks automatically: no calibration keyframe lies within the segment
    or ROLL_SECONDS of it (no keyframe the pre-roll, the post-roll or the offline registration could carry in). The
    gate of the automatic calibration routes that would otherwise compete with the user's keyframes."""
    return not any(start-ROLL_SECONDS <= t <= finish+ROLL_SECONDS for t in keyframe_times)


def _joint_nms(rows, shape):
    """The whole-frame passes of one frame, merged: person roles may disagree across scales, so those
    duplicates are suppressed jointly (the ball is kept apart by an offset)."""
    if not len(rows):
        return rows.numpy()
    categories = (rows[:, 5] == 0).to(rows.dtype)
    separated = rows[:, :4] + categories[:, None]*max(shape[:2])*2
    keep = nms(separated, rows[:, 4], .45)
    return rows[keep].numpy()


def near_pass_adds(detail):
    """Whether the NEAR_PASS pass is a size the whole-frame passes (960 and the detail size) do not run already."""
    return NEAR_PASS not in {960, detail}


def detect_batch(model, frames, detail, near=False, near_min=NEAR_MIN_PX):
    """Whole-frame detection of several consecutive frames of the same size: one model.predict call per
    size pass for all of them (the detector runs them as one batch), then the per-frame merge and NMS
    that detect() does for a single frame. Returns one (N, 6) array per frame, in the frames' order.
    Only the GPU work is shared: every frame's rows come from its own letterboxed image and its own
    suppression. (FP16 batching may move the scores in the last digits; nothing else changes.)
    near: also a NEAR_PASS pass for people only (small scenes), which only fills in (_fill_in) people of at
    least near_min pixels that the other passes missed; the ball is not looked for at that size, where it is a
    few pixels across."""
    if not len(frames):
        return []
    # This checkpoint loses larger foreground players at high input sizes.
    # Keep the 960 whole-frame view and merge an additional detail pass.
    sizes = sorted({960, detail})
    passes = [[] for _ in frames]
    for size in sizes:
        results = model.predict(list(frames), imgsz=size, conf=.10, iou=.45,
                                classes=None, device=0, half=True, verbose=False)
        for collected, result in zip(passes, results):
            collected.append(result.boxes.data.cpu())
    merged = [_joint_nms(torch.cat(collected), frame.shape) for collected, frame in zip(passes, frames)]
    if near and near_pass_adds(detail):
        results = model.predict(list(frames), imgsz=NEAR_PASS, conf=.10, iou=.45,
                                classes=[1, 2, 3], device=0, half=True, verbose=False)
        merged = [_fill_in(rows, result.boxes.data.cpu().numpy(), near_min) for rows, result in zip(merged, results)]
    return merged


def _fill_in(rows, extra, min_height):
    """The rows of the whole-frame passes plus those people of `extra` (the near pass) at least min_height pixels tall
    whose box no person row covers NEAR_FILL_SHARE of (of the smaller box): a person the other passes missed. A near
    row never replaces or doubles a person the other passes found (their boxes would alternate from frame to frame)."""
    extra = np.asarray(extra, np.float32).reshape(-1, 6)
    extra = extra[(extra[:, 3]-extra[:, 1] >= min_height) & (extra[:, 5] != 0)]
    if not len(extra):
        return rows
    people = rows[rows[:, 5] != 0] if len(rows) else np.zeros((0, 6), np.float32)
    area = lambda b: np.maximum(b[..., 2]-b[..., 0], 0)*np.maximum(b[..., 3]-b[..., 1], 0)
    keep = []
    for r in extra:
        if len(people):
            w = np.clip(np.minimum(r[2], people[:, 2])-np.maximum(r[0], people[:, 0]), 0, None)
            h = np.clip(np.minimum(r[3], people[:, 3])-np.maximum(r[1], people[:, 1]), 0, None)
            if (w*h/np.maximum(np.minimum(area(r), area(people)), 1e-6)).max() >= NEAR_FILL_SHARE:
                continue
        keep.append(r)
    if not keep:
        return rows
    return np.concatenate([np.asarray(rows, np.float32).reshape(-1, 6), np.asarray(keep, np.float32)]).astype(rows.dtype if len(rows) else np.float32)


def detect(model, frame, detail, near=False, near_min=NEAR_MIN_PX):
    """Whole-frame detection of one frame (a batch of one)."""
    return detect_batch(model, [frame], detail, near, near_min)[0]


def _colour(person):
    if person['state'] == 'candidate':
        return (255, 255, 255)
    if person['state'] in ('rejected', 'unknown', 'nontarget'):
        return COLORS['UNCERTAIN']
    return COLORS.get(person['role'], COLORS.get(person['team'], COLORS['UNCERTAIN']))


def preview(frame, observation, path, pitch=None, trail=()):
    """Latest processed frame with the pitch boundary, people (display ID, local track, identity
    confidence) and the ball, for the live progress view."""
    image = frame.copy()
    h, w = image.shape[:2]
    if pitch and pitch.get('reliable') and pitch.get('polygon'):
        cv2.polylines(image, [np.int32(np.asarray(pitch['polygon'])*[w, h])], True, COLORS['PITCH'], 2)
    if pitch and pitch.get('status'):
        cv2.putText(image, pitch['status'], (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, .6, COLORS['PITCH'], 2)
    for rejected in observation['rejected']:
        x, y, bw, bh = rejected['box']
        cv2.rectangle(image, (int(x*w), int(y*h)), (int((x+bw)*w), int((y+bh)*h)), (80, 80, 235), 1)
    for person in observation['people']:
        if person['state'] == 'nontarget':
            continue   # ignored people (referees, staff) stay unmarked
        x, y, bw, bh = person['box']
        color = _colour(person)
        a, b = (int(x*w), int(y*h)), (int((x+bw)*w), int((y+bh)*h))
        cv2.rectangle(image, a, b, color, 2 if person['id'] else 1)
        text = f"{person['display']} T{person['track']} {round(person['identityConfidence']*100)}%"
        cv2.putText(image, text, (a[0], max(16, a[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
    points = [(int(x*w), int(y*h)) for _, x, y, _ in trail]
    if len(points) > 1:
        cv2.polylines(image, [np.int32(points)], False, COLORS['BALL'], 1)
    ball = observation['ball']
    if ball['state'] in ('TRACKED', 'MISSING'):
        cx, cy = int(ball['center'][0]*w), int(ball['center'][1]*h)
        cv2.circle(image, (cx, cy), max(7, int(ball['box'][2]*w)), COLORS['BALL'], 2 if ball['state'] == 'TRACKED' else 1)
        if ball.get('predicted'):
            px, py = int(ball['predicted'][0]*w), int(ball['predicted'][1]*h)
            cv2.drawMarker(image, (px, py), COLORS['BALL'], cv2.MARKER_CROSS, 8, 1)
        label = f"BALL {ball['confidence']:.2f}" if ball['state'] == 'TRACKED' else f"BALL PREDICTED {ball['confidence']:.2f} hidden {ball['missingFor']:.1f}s"
        cv2.putText(image, label, (cx+10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, .5, COLORS['BALL'], 1)
    else:
        cv2.putText(image, f"BALL UNKNOWN ({ball.get('phase', 'SEARCHING').lower()})", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, .6, COLORS['BALL'], 1)
    cv2.putText(image, f"{observation['time']:.2f}s | ID, local track, identity confidence", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
    encoded = cv2.imencode('.jpg', cv2.resize(image, (960, round(h/w*960))))[1]
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(encoded.tobytes())
    temporary.replace(path)


class Resources:
    """The detector instances and the appearance embedder of one process, loaded on the first run and kept
    for the next ones (the ~10 s of model loading per run): one detector for the main thread, one for the
    prefetch worker (a predictor keeps state between calls, so each thread has its own), one embedder (a
    stateless ONNX session). Nothing in them depends on a run: the predictor re-applies its arguments on
    every call and its warm-up is a dummy forward pass, so a kept instance gives what a fresh one gives.
    `RESOURCES` is the process's pair; a new Resources() loads afresh (tools/run_segment.py --fresh-models)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.model_path = None
        self.detector = self.worker_detector = self.appearance = None

    @property
    def loaded(self):
        return self.detector is not None

    def load(self, model_path, progress=None):
        """Load once (or again for another checkpoint); later calls return at once."""
        with self.lock:
            if self.detector is None or Path(model_path) != self.model_path:
                if progress is not None:
                    progress({'stage': 'Loading soccer detector and player appearance model', 'progress': 0})
                detector = YOLO(str(model_path))
                if dict(detector.names) != {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}:
                    raise RuntimeError(f'Unexpected detector classes: {detector.names}')
                self.detector, self.worker_detector, self.appearance = detector, YOLO(str(model_path)), Appearance()
                self.model_path = Path(model_path)
            return self

    def release(self):
        with self.lock:
            self.detector = self.worker_detector = self.appearance = self.model_path = None


RESOURCES = Resources()   # one pair of detectors and one embedder per process, kept between runs


class Prefetch:
    """Decodes the segment's frames and runs the whole-frame detector (and the ball tile scans) ahead on a
    worker thread with its own detector instance, so the GPU works on the next frames while the CPU tracks
    the current one. Items come out in frame order and are what the serial loop computed:
    ('roll', index, stamp, frame, None, None, spent) for a pre-roll or post-roll frame (camera motion only)
    and ('frame', index, stamp, frame, rows, tile_rows, spent) for a frame of the segment, tile_rows being
    None on the frames the scans skip; spent holds the worker's seconds per stage. The detections do not
    depend on the tracker, so running them early changes nothing; the passes that do (far tiles, lost-track
    crops, the ball search crop) stay with the tracker on the main thread.

    The ball tile scans do depend on one thing the tracker knows: whether the ball is being followed. The
    main thread writes `ball_locked` after every frame (the ball tracker's phase is LOCKED); the worker
    reads it when it decides a frame's scan, which it does as late as it can, at the moment the hand-over
    has room for that frame, so the word it reads is at most `depth` frames old.

    The NEAR_PASS people pass depends on the tracker too: it runs on small scenes only. The main thread records
    the tracker's small_scene per frame in `small` (note_scene); the worker reads the word of a fixed earlier
    frame (near_wanted), one the main thread has certainly finished, so the pass follows the gate a few frames
    late but identically whatever the timing of the two threads."""

    def __init__(self, video_path, model_path, first_frame, fps, start, finish, roll_until, detail, tiles, cancelled, depth=None, batch=DETECT_BATCH,
                 scan_every=BALL_SCAN_S, model=None, near=False, scale=1.0):
        self.video_path, self.model_path, self.model = video_path, model_path, model   # model: the worker's own detector instance, else loaded here
        self.first_frame, self.fps, self.start, self.finish, self.roll_until = first_frame, fps, start, finish, roll_until
        self.detail, self.tiles, self.cancelled = detail, tiles, cancelled
        self.scale = scale         # tracker.pixel_scale: the ball tiles' size on a low-resolution source
        self.near = bool(near)     # the NEAR_PASS pass is allowed (config nearPass); it runs on small-scene frames only
        self.small = {}            # frame index -> the tracker's small_scene after that frame (note_scene, main thread)
        self.near_frames = 0       # frames detected with the NEAR_PASS pass
        # Consecutive frames of the segment go through the whole-frame detector `batch` at a time (the last
        # group may be shorter); the hand-over holds two groups so that the worker's next batch overlaps the
        # tracking of the current one. A slot is taken per item handed out and given back by the consumer
        # once it has finished with the item.
        self.batch = max(1, int(batch))
        self.depth = 2*self.batch if depth is None else max(1, int(depth))
        self.slots = threading.Semaphore(self.depth)
        self.queue = queue.Queue()
        self.ball_locked = False   # the shared flag: the main thread's word after its latest frame
        self.scan_every = max(1, int(round(float(scan_every)*fps)))   # frames between safeguard scans while locked
        self.scans = 0             # tile scans run
        self.stopped = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._work, name='prefetch', daemon=True)
        self.thread.start()

    def scan_wanted(self, index):
        """Every other frame while the ball tracker is not LOCKED (searching, recovering, occluded: the
        whole frame is where the ball may turn up), once per `scan_every` frames as a safeguard while it is
        (the search crop around the prediction does the work then)."""
        n = index-self.first_frame
        if not self.ball_locked:
            return n % 2 == 0
        return n % self.scan_every == 0

    def note_scene(self, index, small_scene):
        """The main thread's word after frame `index`: the tracker's small_scene."""
        self.small[index] = bool(small_scene)

    def near_wanted(self, first):
        """Whether the batch starting at frame `first` gets the NEAR_PASS pass: the small_scene word of frame
        first-depth-1. When the worker detects that batch it has handed out every item before `first`, and at most
        `depth` of them are still held by the consumer, so frame first-depth-1 is finished and its word written
        (earlier frames, or roll frames, have none: no pass). Reading this frame's word, never a later one, keeps
        the detections independent of how far the consumer happens to be ahead. Never when the detail size is NEAR_PASS
        itself (detect_batch would add nothing), so near_frames counts the frames the extra pass really ran on."""
        return self.near and near_pass_adds(self.detail) and self.small.get(first-self.depth-1, False)

    def _work(self):
        capture = None
        pending = []   # frames of the segment waiting for the batched whole-frame detector: (index, stamp, frame)
        try:
            model = self.model if self.model is not None else YOLO(str(self.model_path))   # a predictor keeps state between calls: one per thread
            capture = cv2.VideoCapture(str(self.video_path))
            capture.set(cv2.CAP_PROP_POS_FRAMES, self.first_frame)
            index = self.first_frame
            while not self.stopped.is_set() and not self.cancelled():
                ok, frame = capture.read()
                if not ok:
                    break
                stamp = float(capture.get(cv2.CAP_PROP_POS_MSEC))/1000
                if stamp <= 0 and index:
                    stamp = index/self.fps
                if stamp < self.start-1e-6 or (stamp >= self.finish and self.roll_until is not None and stamp <= self.roll_until+.15):
                    self._flush(model, pending)   # the frames before this one go out first: the order is the video's
                    self._put(('roll', index, stamp, frame, None, None, {'detect': 0.0, 'ballTiles': 0.0}))
                    index += 1
                    continue
                if stamp >= self.finish:
                    break
                pending.append((index, stamp, frame))
                index += 1
                if len(pending) >= self.batch:
                    self._flush(model, pending)
            if not self.cancelled():
                self._flush(model, pending)
        except BaseException as error:   # noqa: BLE001 - handed to the consumer
            self.error = error
        finally:
            if capture is not None:
                capture.release()
            self._put(None)

    def _flush(self, model, pending):
        """Detect the pending frames in one batch and hand them on, one item per frame, in order. The batch's
        seconds are shared out equally per frame, so the per-frame timing adds up to what the worker spent."""
        if not pending:
            return
        clock = time.perf_counter
        if self.stopped.is_set():
            pending.clear()
            return
        t0 = clock()
        near = self.near_wanted(pending[0][0])
        detections = detect_batch(model, [frame for _, _, frame in pending], self.detail, near, NEAR_MIN_PX*self.scale)
        self.near_frames += len(pending) if near else 0
        each = (clock()-t0)/len(pending)
        for (index, stamp, frame), rows in zip(pending, detections):
            if not self._room():
                break
            spent = {'detect': each, 'ballTiles': 0.0}
            tile_rows = None
            if self.tiles and self.scan_wanted(index):   # decided now, with the main thread's latest word
                t0 = clock()
                tile_rows = ball_tiles(model, frame, self.scale)
                spent['ballTiles'] = clock()-t0
                self.scans += 1
            self.queue.put(('frame', index, stamp, frame, rows, tile_rows, spent))
        pending.clear()

    def _room(self):
        """Wait for a free slot in the hand-over; False once the consumer has closed it."""
        while not self.stopped.is_set():
            if self.slots.acquire(timeout=.1):
                return True
        return False

    def _put(self, item):
        if item is None:
            self.queue.put(None)   # the end: no slot needed, and never blocked
        elif self._room():
            self.queue.put(item)

    def __iter__(self):
        while True:
            item = self.queue.get()
            if item is None:
                if self.error is not None:
                    raise self.error
                return
            yield item
            self.slots.release()   # the consumer is done with it (and has updated the shared flag)

    def close(self):
        self.stopped.set()
        try:
            while True:
                self.queue.get_nowait()
        except queue.Empty:
            pass
        self.thread.join(timeout=10)


def run(video_path, output, config, progress, cancelled, previous=None, resources=None):
    """previous: saved identities of an earlier run of this video, to continue its global identities.
    resources: the Resources (detectors, embedder) to run with; the process's RESOURCES by default, loaded on
    the first run and kept for the next."""
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    gpu = gpu_info()
    meta = metadata(video_path)
    width, height, fps = meta['width'], meta['height'], meta['fps']
    start = float(config.get('start', 0))
    finish = min(meta['duration'], start+float(config.get('seconds', 60)))
    model_path = get_model()
    shared = resources if resources is not None else RESOURCES
    if not shared.loaded:
        progress({'stage': 'Loading soccer detector and player appearance model', 'progress': 0, 'gpu': gpu})
    shared.load(model_path)
    model, appearance = shared.detector, shared.appearance
    tracker = SoccerTracker(width, height, config_from(config), BotSortTracker(fps), appearance.embed, previous, fps)
    # Keyframes just outside the segment are not lost: the frames from such a keyframe to the start are processed
    # for camera motion and pitch evidence only (pre-roll), and after the end up to such a keyframe (post-roll).
    keyframe_times = sorted(float(c['time']) for c in (config.get('calibrations') or []))
    tracker.auto_landmarks = auto_landmarks(keyframe_times, start, finish)
    roll_from = min((t for t in keyframe_times if start-ROLL_SECONDS <= t < start), default=None)
    roll_until = max((t for t in keyframe_times if finish <= t <= finish+ROLL_SECONDS), default=None)
    first_frame = int(round((start if roll_from is None else roll_from)*fps))
    frames = []
    torch.cuda.reset_peak_memory_stats()
    (output/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    log = (output/'events.log').open('w', encoding='utf-8')
    index, last_notice, logged, pitch, far_passes, ball_searches, rolled, crop_passes = first_frame, -10, 0, None, 0, 0, 0, 0
    timing = Counter()   # seconds per stage, for summary.timing (detect and ballTiles: the worker's, overlapping the tracking)
    clock = time.perf_counter
    ahead = Prefetch(video_path, model_path, first_frame, fps, start, finish, roll_until, int(config.get('imageSize', 1280)),
                     bool(config.get('ballTiles', False)), cancelled, batch=int(config.get('detectBatch', DETECT_BATCH)),
                     scan_every=float(config.get('ballScanEvery', BALL_SCAN_S)), model=shared.worker_detector,
                     near=bool(config.get('nearPass', True)), scale=tracker.pixel_scale)
    debug = bool(config.get('debug', True))
    try:
        for kind, index, stamp, frame, rows, tile_rows, spent in ahead:
            if cancelled():
                break
            timing['detect'] += spent['detect']
            timing['ballTiles'] += spent['ballTiles']
            if kind == 'roll':
                # Pre-roll or post-roll: camera motion and pitch evidence only, no detections, nothing emitted.
                t0 = clock()
                tracker.roll(frame, stamp)
                timing['roll'] += clock()-t0
                rolled += 1
                if stamp < start and not frames:
                    tracker.offset = len(tracker.frame_log)
                elapsed = time.perf_counter()-started
                if elapsed-last_notice >= 1:
                    which = f'before the start, from the keyframe at {roll_from:.1f} s' if stamp < start else f'after the end, to the keyframe at {roll_until:.1f} s'
                    progress({'stage': f'Carrying the calibration through the frames {which} (camera motion only)', 'progress': 1.0 if stamp >= finish else 0.0,
                              'videoTime': stamp, 'elapsed': round(elapsed, 1), 'frames': len(frames), 'gpu': gpu})
                    last_notice = elapsed
                continue
            whole = rows
            band = tracker.far_band()
            if band is not None:
                far_passes += 1
                t0 = clock()
                rows = merge_far(rows, far_tiles(model, frame, band))
                timing['farTiles'] += clock()-t0
            windows = tracker.lost_windows()
            if windows:
                crop_passes += 1
                t0 = clock()
                rows = merge_crops(rows, local_crops(model, frame, windows))
                timing['localCrops'] += clock()-t0
            window = tracker.ball.search_window()
            crop_rows = []
            if window is not None and config.get('ballSearch', True):
                ball_searches += 1
                t0 = clock()
                crop_rows = ball_crop(model, frame, window, tracker.pixel_scale)
                timing['ballCrop'] += clock()-t0
            else:
                window = None
            ball_rows, passes = ball_candidates(whole, tile_rows, crop_rows, width, height)
            t0 = clock()
            observation, _, _ = tracker.step(frame, stamp, rows[rows[:, 5] != 0], ball_rows)
            timing['tracking'] += clock()-t0
            # The worker's words: it scans the ball tiles only while the ball is not followed by a certain lock (plus its
            # safeguard), and adds the near pass on small scenes.
            ahead.ball_locked = tracker.ball.scan_locked
            ahead.note_scene(index, tracker.small_scene)
            observation = {'time': round(stamp, 6), 'frame': index, **observation}
            if debug:
                observation['ballPasses'] = passes | {'tiles': tile_rows is not None,
                                                      'crop': None if window is None else [round(window[0]/width, 4), round(window[1]/height, 4),
                                                                                           round(window[2]/width, 4), round(window[3]/height, 4)]}
            frames.append(observation)
            tracker.emitted = len(frames)
            pitch = observation['pitch'] or pitch
            # Every identity, role and ball decision is logged; the key ones also go to the console.
            for event in tracker.events[logged:]:
                log.write(f"[{event['time']:.2f}s] {event['kind'].upper()}\n{event['message']}\n\n")
                if event['kind'] in ('reid', 'swap-corrected', 'sanity', 'role', 'pitch'):
                    print(f"[{event['time']:.2f}s] {event['message']}", flush=True)
            logged = len(tracker.events)
            elapsed = time.perf_counter()-started
            if elapsed-last_notice >= 1:
                preview(frame, observation, output/'preview.jpg', pitch, tracker.ball.trajectory)
                counts = tracker.ids.summary()
                progress({'stage': 'Tracking match participants and global identities', 'progress': (stamp-start)/(finish-start),
                          'videoTime': stamp, 'elapsed': round(elapsed, 1), 'frames': len(frames),
                          'processingFPS': round(len(frames)/elapsed, 1), 'people': len(observation['people']),
                          'identities': {k: len(v) for k, v in counts.items()}, 'warning': tracker.keyframes.warning})
                last_notice = elapsed
    finally:
        ahead.close()
        log.close()
    if not frames:
        raise RuntimeError('No video frames were processed.')
    online_events = {id(e) for e in tracker.events}
    # The pitch registration comes first: the tracklet passes then compare positions in metres where they exist.
    progress({'stage': 'Registering the pitch over the whole segment and remapping the tracks'})
    t0 = clock()
    propagated, timeline = finalize_pitch(frames, tracker)
    timing['offlinePitch'] += clock()-t0
    progress({'stage': 'Refining tracklets over the whole segment (split mixed tracks, reconnect fragments, assign the rest)'})
    t0 = clock()
    identity_report = finalize_identities(frames, tracker)
    timing['offlineIdentities'] += clock()-t0
    progress({'stage': 'Decoding the final ball trajectory over the whole segment'})
    t0 = clock()
    ball_report = finalize_ball(frames, tracker, timeline)
    timing['offlineBall'] += clock()-t0
    progress({'stage': 'Writing global identities and tracking data'})
    t0 = clock()
    report = build_report(frames, tracker, meta, gpu, config, time.perf_counter()-started, cancelled(), previous, ball_report, identity_report)
    report['summary']['pitch']['propagatedFrames'] = propagated
    report['summary']['pitch']['farTilePasses'] = far_passes
    report['summary']['detection'] = dict(tracker.detection) | {'farTilePasses': far_passes, 'localCropPasses': crop_passes,
                                                                 'ballTileScans': ahead.scans, 'detectBatch': ahead.batch,
                                                                 'nearPassFrames': ahead.near_frames}
    report['summary']['profile'] = run_profile(frames, tracker, ahead.near)
    report['summary']['pitch']['registration'] = tracker.registration
    report['summary']['ball']['localSearches'] = ball_searches
    timing['report'] += clock()-t0
    report['summary']['timing'] = stage_timing(timing, len(frames), time.perf_counter()-started, rolled)
    snapshot = tracker.snapshot() | {'run': output.name, 'start': start, 'end': frames[-1]['time']}
    (output/'identities.json').write_text(json.dumps(snapshot, separators=(',', ':')), encoding='utf-8')
    (output/'registration.json').write_text(json.dumps(registration_record(tracker, meta, config), separators=(',', ':'), default=_json_default),
                                            encoding='utf-8')
    # A numpy number left in a debug field must not cost the run its result (the default is consulted for nothing else).
    (output/'result.json').write_text(json.dumps(report, separators=(',', ':'), allow_nan=False, default=_json_default), encoding='utf-8')
    # The offline passes' events (tracklets, registration, ball path) go to the log too, after the online ones.
    with (output/'events.log').open('a', encoding='utf-8') as log:
        for event in tracker.events:
            if id(event) not in online_events:
                log.write(f"[{event['time']:.2f}s] {event['kind'].upper()} (offline)\n{event['message']}\n\n")
    progress({'stage': report['status'], 'progress': 1, 'summary': report['summary'], 'teams': report['teams']})
    return report


def run_profile(frames, tracker, near):
    """summary.profile: the run's gates. A run is a small scene when at least half its frames are (the offline passes
    decide so from frames[].profile themselves); autoLandmarks is the per-run gate of the automatic calibration.
    smallSceneChanges counts the frames where the gate differs from the frame before (the gate latches, so at most one
    in a run) and smallSceneFrom is the time of the first small-scene frame (None in a run without one)."""
    flags = [bool((f.get('profile') or {}).get('smallScene')) for f in frames]
    small = sum(flags)
    return {'smallSceneFrames': small, 'smallSceneShare': round(small/max(1, len(frames)), 3), 'smallSceneRun': bool(frames) and small >= .5*len(frames),
            'smallSceneChanges': sum(1 for a, b in zip(flags, flags[1:]) if a != b),
            'smallSceneFrom': next((f.get('time') for f, on in zip(frames, flags) if on), None),
            'autoLandmarks': bool(tracker.auto_landmarks), 'pixelScale': round(float(tracker.pixel_scale), 4), 'nearPass': bool(near)}


def stage_timing(timing, frames, elapsed, rolled=0):
    """Milliseconds per processed frame by online stage, the offline passes in seconds, and the total."""
    online = ('detect', 'farTiles', 'localCrops', 'ballTiles', 'ballCrop', 'tracking')
    out = {f'{k}MsPerFrame': round(1000*timing[k]/max(1, frames), 1) for k in online}
    out['onlineMsPerFrame'] = round(1000*sum(timing[k] for k in online)/max(1, frames), 1)
    out['offlineSeconds'] = {k: round(timing[k], 2) for k in ('offlineIdentities', 'offlinePitch', 'offlineBall', 'report')}
    out['rollFrames'] = rolled
    out['rollSeconds'] = round(timing['roll'], 2)
    out['elapsedSeconds'] = round(elapsed, 2)
    return out


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'not serialisable: {type(value).__name__}')


def registration_record(tracker, meta, config):
    """Everything the offline registration used, saved per run (registration.json) so that the hold-out
    tests (tools/holdout.py) can rebuild the timeline without re-running the detector: the per-frame camera
    motion log, every validated anchor with the keyframe its chain started from, and the keyframes."""
    applied = {round(float(k['time']), 3): k for k in tracker.keyframe_log}
    keyframes = []
    for c in sorted(config.get('calibrations', []) or [], key=lambda c: c['time']):
        k = applied.get(round(float(c['time']), 3))
        keyframes.append({'time': float(c['time']), 'index': k['index'] if k else None, 'rms': round(float(k['rms']), 3) if k and k['rms'] is not None else None,
                          'observability': k['observability'] if k else None, 'points': c['points']})
    return {'version': 1, 'width': meta['width'], 'height': meta['height'], 'fps': meta['fps'], 'offset': getattr(tracker, 'offset', 0),
            'emitted': getattr(tracker, 'emitted', len(tracker.frame_log)), 'size': tracker.keyframes.size.to_json(),
            'frames': [{'t': round(e['t'], 6), 'dt': round(e['dt'], 6), 'M': None if e['M'] is None else np.asarray(e['M'], np.float64).ravel().tolist(),
                        'reliable': bool(e['reliable']), 'cut': bool(e['cut']), 'quality': e.get('quality') or 'none'} for e in tracker.frame_log],
            'anchors': [{'index': a.index, 'time': round(a.time, 6), 'H': np.asarray(a.H, np.float64).ravel().tolist(), 'observability': float(a.observability),
                         'source': a.source, 'root': a.root, 'details': a.details} for a in tracker.anchors],
            'keyframes': keyframes}


def finalize_pitch(frames, tracker):
    """Offline passes 2 and 3 (soccer/registration.py): every validated anchor of the run (landmark
    calibrations, painted markings that confirmed the model, automatic midfield anchors) is carried
    forward and backward through the recorded camera motion, the two directions are blended, and the
    pitch positions of people and the ball are recomputed on every frame from the final geometry. The
    image tracks and identities are not touched. Returns the number of frames that gained a pitch model."""
    log = tracker.emitted_log()
    if len(log) != len(frames):
        return 0, None
    W, H = tracker.width, tracker.height
    size = tracker.keyframes.size
    # The chain runs over the whole log, pre-roll and post-roll included; the segment is the slice.
    offset = tracker.offset
    timeline = registration.build_timeline(tracker.frame_log, tracker.anchors, W, H, size)[offset:offset+len(frames)]
    online = [bool(f.get('calibrated')) for f in frames]
    metrics = registration.remap_frames(frames, timeline, W, H, size)
    filled = 0
    for frame, entry, was in zip(frames, timeline, online):
        Hm, confidence = entry['H'], entry['confidence']
        if Hm is None or not frame.get('calibrated'):
            continue
        if not was:
            filled += 1
            frame['pitchConfidence'] = round(float(confidence), 3)
            frame['pitchMode'] = 'automatic' if confidence >= pitch_model.HIGH else 'robust'
        if frame.get('pitch') and (not was or frame['pitch'].get('level') == pitch_model.LOW_LEVEL):
            outline = pitch_model.field.outline(Hm, size, W, H)
            if outline:
                p = frame['pitch']
                p['polygon'] = [[round(x, 4), round(y, 4)] for x, y in outline]
                p['source'], p['confidence'], p['level'] = 'model', round(float(confidence), 3), pitch_model.levels(confidence)
                p['mode'] = 'automatic' if confidence >= pitch_model.HIGH else 'robust'
                how = {'carried': f"CARRIED {entry['direction'] or ''} FROM THE ANCHOR AT FRAME {entry['anchor']}",
                       'interpolated': f"BLENDED FROM THE ANCHORS AROUND IT (nearest: frame {entry['anchor']})"}.get(entry['source'], entry['source'].upper())
                p['status'] = f'PITCH CONFIDENCE: {confidence:.2f} · OFFLINE REGISTRATION: {how.strip()} · observability {entry["observability"]:.2f}'
                p['edges'] = pitch_model.field.side_status(Hm, size, W, H)
                p['evidence'] = [f"filled by the offline registration pass: {entry['source']} ({entry['direction'] or 'at the anchor'}), "
                                 f"anchor {entry['age']:.1f}s away, observability {entry['observability']:.2f}"
                                 + (f", forward/backward disagreement {entry['disagreement']:.1f} m" if entry['disagreement'] is not None else '')]
    bridged = registration.bridge_ball(frames, W)
    tracker.events += registration.warnings(frames)
    tracker.events.sort(key=lambda e: e['time'])
    # A section that asked for confirmation online is resolved when the registration now covers it.
    times = [f['time'] for f in frames]
    covered = lambda s: all(f.get('calibrated') for f, t in zip(frames, times) if s['start']-1e-6 <= t <= s['end']+1e-6)
    tracker.pitch_issues()   # closes a section still open at the end
    tracker.sections = [s for s in tracker.sections if not covered(s)]
    ticks = [f['pitch'] for f in frames if f.get('pitch')]
    curvatures = [p['curvature'] for p in ticks if p.get('curvature') is not None]
    tracker.registration = registration.summarize(timeline, tracker.anchors, offset, size) | metrics | {
        'propagatedFrames': filled, 'ballFramesBridged': bridged, 'registrations': dict(getattr(tracker, 'registrations', {})),
        'lens': lens.summarize(getattr(tracker, 'lens_samples', [])),
        'lensCurvature': round(float(np.median(curvatures)), 2) if curvatures else None,
        'vanishingPointShare': round(sum(1 for p in ticks if p.get('vanishing'))/len(ticks), 3) if ticks else 0.0,
        'mappingWarnings': sum(1 for e in tracker.events if e['kind'] == 'calibration')}
    return filled, timeline


def finalize_identities(frames, tracker):
    """Offline tracklet refinement (soccer/tracklets.py): local tracks are cut where their identity label
    changed, and fragments without a confident player identity are reconnected to the dormant identities
    around them in time. Relabels the frames in place; returns the report, or None."""
    log = tracker.emitted_log()
    if len(log) != len(frames) or not frames:
        return None
    motions = [(e['M'], e['reliable'], e['cut']) for e in log]
    size = tracker.keyframes.size
    report = tracklets.refine(tracker.ids, frames, motions, tracker.width, tracker.height, size)
    # The fragments still anonymous after the refinement: one global assignment with mutual exclusion.
    assigned = tracklets.assign_fragments(tracker.ids, frames, motions, tracker.width, tracker.height, size)
    # A fragment identified by the assignment hands its identity to the anonymous stretches of its own track.
    continued = tracklets.continue_same_track(tracker.ids, frames, motions, tracker.width, tracker.height, size)
    for key in ('connected', 'sameTrack', 'framesRestored'):
        report[key] = report.get(key, 0)+continued[key]
    report['connections'] = report.get('connections', [])+continued['connections']
    # Identities that may be one person: proposed for review in roster.mergeProposals, never merged here.
    proposals = tracklets.merge_proposals(tracker.ids, frames, motions, tracker.width, tracker.height, size)
    # A later identity that only continues an earlier one across a short occlusion is merged into it; the other
    # proposals stay for review.
    merged = tracklets.apply_merges(tracker.ids, proposals['proposals'], frames[-1]['time'])
    applied = {p['later'] for p in proposals['proposals'] if p.get('applied')}
    report['mergeProposals'] = proposals['proposals']
    report['mergeProposalCounts'] = {k: proposals[k] for k in ('pairsCompared', 'rejectedByAppearance', 'rejectedByContinuity', 'interleaved')}
    report['merged'] = merged['merged']
    report['assigned'], report['framesAssigned'] = assigned['accepted'], assigned['framesAssigned']
    if 'smallAccepted' in assigned:
        report['smallAssigned'] = assigned['smallAccepted']   # small-scene runs: fragments the assignment placed on position
    report['assignment'] = {k: assigned[k] for k in ('proposed', 'accepted', 'rejectedByLead', 'rejectedByScore', 'rounds')}
    report['events'] = report['events']+assigned['events']+continued['events']+[e for e in proposals['events'] if e.get('playerId') not in applied]+merged['events']
    tracker.events += report['events']
    tracker.events.sort(key=lambda e: e['time'])
    return report


def finalize_ball(frames, tracker, timeline):
    """Offline pass 4 (soccer/ball_path.py): the final ball trajectory decoded from every frame's candidates
    and an explicit "not visible" node over the whole segment. Rewrites each frame's `ball` (the causal
    first pass stays as `firstPass`), maps it through the registration's final homographies, and logs the
    first-pass sightings the final path leaves out. Returns the decoder's report, or None."""
    log = tracker.emitted_log()
    if len(log) != len(frames) or not frames:
        return None
    W, H = tracker.width, tracker.height
    size = tracker.keyframes.size

    def to_pitch(i, point):
        entry = timeline[i] if timeline is not None and i < len(timeline) else None
        if entry is None or entry.get('H') is None:
            return None
        p, _, _ = registration.position(entry['H'], size, point, entry['confidence'])
        return p

    motions = [(e['M'], e['reliable'], e['cut']) for e in log]
    report = ball_path.finalize(frames, motions, W, H, tracker.ball.fps, to_pitch)
    for i, f in enumerate(frames):
        b = f['ball']
        if b.get('pitch') and timeline is not None and i < len(timeline) and timeline[i].get('H') is not None:
            _, c, sigma = registration.position(timeline[i]['H'], size, (b['center'][0]*W, min(1.0, b['center'][1]+b['box'][3]/2)*H), timeline[i]['confidence'],
                                                None, timeline[i].get('disagreement'))
            if c is not None:
                b['pitchConfidence'] = c
            if sigma is not None:
                b['pitchSigma'] = [round(sigma[0], 1), round(sigma[1], 1)]
    tracker.events += report['events']
    tracker.events.sort(key=lambda e: e['time'])
    return report


def build_report(frames, tracker, meta, gpu, config, elapsed, cancelled, previous, ball_report=None, identity_report=None):
    tracks, people, identified = {}, 0, 0
    for frame in frames:
        for person in frame['people']:
            info = tracks.setdefault(person['track'], {'firstSeen': frame['time'], 'lastSeen': frame['time'], 'observations': 0, 'ids': {}})
            info['lastSeen'] = frame['time']
            info['observations'] += 1
            if person['id']:
                info['ids'][person['id']] = info['ids'].get(person['id'], 0)+1
            if person['state'] not in ('rejected', 'unknown', 'nontarget'):   # ignored people are nobody's identity by design
                people += 1
                identified += person['id'] is not None
    match = tracker.ids.summary()
    events = tracker.events
    kinds = Counter(event['kind'] for event in events)
    count = lambda team: sum(1 for p in match['players'] if p['team'] == team)
    states = Counter(f['ball']['state'] for f in frames)
    seen = [f['ball']['confidence'] for f in frames if f['ball']['state'] == 'TRACKED']
    share = lambda n: round(n/len(frames), 3)
    phases = Counter(f['ball'].get('phase', 'SEARCHING') for f in frames)
    jumps = ball_path.shown_jumps(frames, meta['width'], meta['height'])
    first_pass = {'reacquisitions': sum(1 for e in events if e['kind'] == 'ball' and e['message'].startswith('BALL REACQUIRED')),
                  'lost': sum(1 for e in events if e['kind'] == 'ball' and e['message'].startswith('BALL LOST')),
                  'switchWarnings': sum(1 for e in events if e['kind'] == 'ball' and e['message'].startswith('BALL SWITCH WARNING')),
                  'tracks': tracker.ball.track}
    if ball_report:
        first_pass.update(ball_report['firstPass'])
    ball = {'trackedShare': share(states['TRACKED']), 'missingShare': share(states['MISSING']), 'unknownShare': share(states['UNKNOWN']),
            'meanConfidence': round(sum(seen)/len(seen), 3) if seen else None,
            'tracks': ball_report['final']['runs'] if ball_report else tracker.ball.track,
            'teleports': jumps['teleports'], 'footSwitches': jumps['footSwitches'],
            'longestUnknownSeconds': ball_report['final']['longestUnknownSeconds'] if ball_report else None,
            'droppedFirstPassRuns': len(ball_report['dropped']) if ball_report else 0, 'firstPass': first_pass,
            **({k: ball_report['final'].get(k, 0) for k in ('staticRunsDropped', 'longUnseenLinks', 'stationaryLinks', 'markingNodes', 'lockAudits')} if ball_report else {}),
            'reacquisitions': first_pass['reacquisitions'], 'lost': first_pass['lost'],
            'phases': {k: share(v) for k, v in phases.items()}, 'rejectedCandidates': dict(tracker.ball.rejections.most_common())}
    nontarget = [p for f in frames for p in f['people'] if p.get('state') == 'nontarget']
    roles = {'playerIdentitiesWithdrawn': len(match.get('withdrawn', [])),
             'nonTargetTracks': len({p['track'] for p in nontarget}),
             'goalkeepersIdentified': sum(1 for e in events if e['kind'] == 'role' and e['message'].startswith('GOALKEEPER IDENTIFIED')),
             'goalkeepersRevoked': sum(1 for e in events if e['kind'] == 'role' and e['message'].startswith('GOALKEEPER REVOKED')),
             'goalkeeperCandidatesDropped': sum(1 for e in events if e['kind'] == 'role' and e['message'].startswith('GOALKEEPER CANDIDATE DROPPED')),
             'goalkeeperTeams': sum(1 for e in events if e['kind'] == 'role' and e['message'].startswith('GOALKEEPER TEAM')),
             'mergedIdentities': len(match['retired'])}
    # For the 2D match view: how much of the run has pitch coordinates, and which team defends which end
    # (pitch x = 0 is the left goal line as the calibration names it, which is the image's left).
    sides = tracker.ids.goal_sides()

    def defends(side):
        votes = sides.get(side) or {}
        total = sum(votes.values())
        if total < 15:
            return None
        team, n = max(votes.items(), key=lambda item: item[1])
        return team if n/total >= .75 else None
    calibrated = sum(1 for f in frames if f.get('calibrated'))
    size = getattr(getattr(tracker, 'keyframes', None), 'size', None) or pitch_model.field.PitchSize()
    tactical = {'coordinates': 'normalized pitch position of the feet: x 0..1 from the left goal line to the right goal line, '
                               f'y 0..1 from the far touchline to the near touchline ({size.length:.0f} x {size.width:.0f} m), from the landmark calibration carried through camera motion',
                'calibratedFrames': calibrated, 'calibratedShare': share(calibrated), 'goalEnds': {'left': defends('left'), 'right': defends('right')},
                'pitchSize': size.to_json(), 'coordinateSources': dict(Counter(f.get('pitchCoordinates') or 'none' for f in frames))}
    ticks = [f['pitch'] for f in frames if f.get('pitch')]
    modes = Counter(f.get('pitchMode', 'automatic') for f in frames)
    levels_ = Counter(p.get('level', 'LOW') for p in ticks)
    pitch_issues = tracker.pitch_issues() if hasattr(tracker, 'pitch_issues') else []
    pitch = {'meanConfidence': round(sum(p.get('confidence', 0) for p in ticks)/len(ticks), 3) if ticks else 0.0,
             'levels': {k: round(v/len(ticks), 3) for k, v in levels_.items()} if ticks else {},
             'modes': {k: share(v) for k, v in modes.items()}, 'size': size.to_json(),
             'sectionsNeedingConfirmation': len(pitch_issues),
             'modelEvents': sum(1 for e in events if e['kind'] == 'pitch')}
    summary = {'processedFrames': len(frames), 'localTrackSegments': len(tracks),
               'globalIdentities': {'teamA': count('A'), 'teamB': count('B'), 'goalkeepers': sum(1 for g in match['goalkeepers'] if not g.get('revoked')),
                                    'referees': len(match['referees']),   # always 0: referees are not tracked as identities
                                    'merged': len(match['retired']), 'excluded': len(match.get('excluded', [])), 'withdrawn': len(match.get('withdrawn', []))},
               'nonTarget': {'tracks': roles['nonTargetTracks'], 'frames': len(nontarget),
                             'hints': dict(Counter(p.get('hint') or 'REFEREE_LIKE' for p in nontarget))},
               'remembered': sorted(p['id'] for key in ('players', 'goalkeepers', 'referees', 'excluded') for p in match.get(key, []) if p.get('manual')), 'corrections': 0,
               'reidentifications': kinds.get('reid', 0), 'deferredDecisions': kinds.get('deferred', 0), 'newIdentities': kinds.get('new-identity', 0),
               'swapCorrections': kinds.get('swap-corrected', 0), 'crossingWarnings': kinds.get('swap-uncertain', 0), 'sanityWarnings': kinds.get('sanity', 0),
               'identifiedShare': round(identified/people, 3) if people else 0.0, 'rejectedDetections': tracker.rejected_counts,
               'cameraCuts': tracker.cuts, 'meanVisiblePeople': round(sum(len(f['people']) for f in frames)/len(frames), 1),
               'ball': ball, 'roles': roles, 'goalSides': tracker.ids.goal_sides(), 'goalkeeperSlots': tracker.ids.keeper_slots(), 'pitch': pitch,
               'identity': tracklets.metrics(tracker.ids, frames, events, identity_report or {}) if hasattr(tracker.ids, 'encounters') else None,
               'elapsedSeconds': round(elapsed, 2), 'processingFPS': round(len(frames)/elapsed, 2),
               'peakGPUMemoryGB': round(torch.cuda.max_memory_allocated()/1024**3, 2) if torch.cuda.is_available() else 0.0}
    report = {'version': 3, 'video': meta, 'gpu': gpu, 'config': config, 'frames': frames, 'match': match, 'tracks': tracks,
            'events': events, 'issues': tracker.issues[-500:], 'pitchIssues': pitch_issues, 'teams': tracker.model.to_json(), 'tactical': tactical,
            'continuedFrom': (previous or {}).get('run'), 'status': 'cancelled' if cancelled else 'complete', 'summary': summary,
            'labels': {p['id']: long_id(p['id']) for group in match.values() for p in group},
            'limitations': ['Global IDs (A-07, GK-1) are tracking identities, not jersey numbers or names. Goalkeepers are shown by team (GK-A, GK-B).',
                            'Referees, assistant referees, coaches and staff are not tracked as identities: they are ignored people (NON_TARGET) with a local box only, used to know when someone overlaps a player, never shown in the 2D view.',
                            'Identity is kept only when the evidence is clear; uncertain people stay unidentified rather than guessed.',
                            'Roles come from about ten seconds of detector, kit and position evidence. A referee-like reading never changes a confirmed player; only an identity that was never an established player and showed the same referee-like person from its start is withdrawn (its observations become NON_TARGET).',
                            'OSNet is a general pedestrian model: same-kit teammates can look alike, so some returns stay unresolved.',
                            'Pitch coordinates exist where the pitch model is trusted: anchored by a calibration keyframe, carried through the camera motion (also backwards, in the offline pass) and confirmed by painted lines; on sustained low confidence a section asks for confirmation instead of guessing.',
                            'People you corrected in the review page are remembered by appearance: a run recognises them only when they look clearly like the person you corrected.',
                            'Tracked people whose identity stayed unresolved are real observations: the 2D view shows them as ? markers of their team, never as identities, and the roster counts them as anonymous.',
                            'Every roster slot not observed on a frame is classified from the camera view: expected in frame but not seen, off-screen, or unknown; this is evidence for the review, never a reason to remove or invent anyone.',
                            'Where a roster slot is not observed, the timeline carries an estimate with its kind and sigma: I (interpolated across a gap of at most a second), P (predicted: the straight line between two observations further apart, or, where only one side is observed, the last position moved with the visible teammates; the sigma says how far to trust it) or U (unknown: nothing supportable, no coordinates). Estimates are never observations and the 2D view draws them differently.',
                            'The ball is one physical identity (BALL-1). Its final trajectory is decoded offline over the whole segment from every frame\'s candidates and an explicit "not visible" node: a sighting that no plausible trajectory leads into or out of (a boot, a sock, a spot) is left out, short occlusions on a trajectory are bridged (MISSING), and everything else is UNKNOWN rather than a guess. The causal first pass is kept per frame as firstPass. No ball accuracy is claimed.']}
    # Display labels (A-01, B-07) are regenerated from the confirmed roster, apart from the internal identities
    # (soccer/labels.py): numbered without gaps, at most the outfield slots active at a time, the rest unconfirmed.
    labelling = labels.assign(report)
    if any(c['unconfirmed'] for c in labelling['counts'].values()):
        events.append({'time': round(frames[-1]['time'], 3), 'kind': 'labels', 'track': None, 'playerId': None,
                       'message': '\n'.join(['PLAYER LABELS']+labels.describe(labelling))})
    # The active roster and, per frame, what the camera sees of the pitch and where each unobserved slot stands
    # with respect to it: pure functions of the result, re-derived whenever a correction is applied.
    report['roster'] = roster.build(report)
    report['roster']['mergeProposals'] = list((identity_report or {}).get('mergeProposals') or [])
    summary['roster'] = roster.summary(report['roster'])
    # Where every slot probably is on every frame (observed / interpolated / predicted / unknown, with a
    # sigma), from the whole segment; the visibility classification then uses these estimates.
    pitch_size = report['tactical']['pitchSize']
    report['timeline'] = imputation.estimate(report, float(pitch_size['length']), float(pitch_size['width']))
    summary['timeline'] = {'method': report['timeline']['method'], 'kinds': report['timeline']['kinds'], 'slots': len(report['timeline']['ids'])}
    summary['visibility'] = visibility.annotate(report)
    return report
