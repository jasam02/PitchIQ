"""Exact source frames for the coaching report's screenshots, as cached JPEGs of raw pixels.

Why by frame index. Every evidence record carries frames[i].frame, the absolute decode index the pipeline counted
from its one CAP_PROP_POS_FRAMES seek (pipeline.py), so a screenshot must be that decode, not "the frame near this
time". The server audit measured it on the three uploads (OpenCV 4.12 FFMPEG, h264 1920 x 1080, constant 30 fps), 60
random targets each against a sequential decode from frame 0:
- CAP_PROP_POS_FRAMES = n: 60/60 exact on every video (and identical pixels: mean difference 0.00, audit/calibration);
- CAP_PROP_POS_MSEC = (n + .5)/fps (a time in the middle of the frame): 59/60 land on frame n+1, because a time seek
  rounds to the nearest frame. So a time is never used to seek here.
After each read the decoder's own time (CAP_PROP_POS_MSEC, the pipeline's frames[i].time) is checked against the
expected time of that frame, within half a frame: a container whose seek lands elsewhere is refused frame by frame
instead of returning the wrong picture. As in the pipeline, a decoder that reports 0 ms after frame 0 is taken at
n / fps.

Why sorted, grab or seek. A random seek and read costs 72-150 ms (median; p90 up to 324 ms on the 754 MB amateur
file) against about 3.3 ms per frame decoded in order (301-348 fps). The frames are therefore read in increasing
order from one capture, decoding forward (grab) when the next one is at most GRAB_AHEAD frames ahead (45 frames at
3.3 ms is about one seek) and seeking otherwise: 24 evidence frames took 1.08 s instead of 2.15 s on broadcast, 2.27
instead of 3.20 s on amateur and 1.93 instead of 4.35 s on Veo (server audit). Measured here with extract() (24 frames
per upload, 20 spread over the run and 4 near neighbours, written at 640 and 1280 px): 1.44 s broadcast, 4.68 s
amateur, 2.19 s Veo, against 3.75 / 8.99 / 5.89 s with one capture per frame; a second call from the cache 0.01 s;
on each upload 9 of 9 targets (seek and grab paths) identical to the sequential decode, mean difference 0.00.

Why JPEGs of raw pixels. The frames are resized to 640 or 1280 px wide (the report's half-width column and a full
width; never enlarged) and stored as baseline JPEG quality 85, which the PDF passes through as /DCTDecode (median
41-52 kB at 640 px and 118-163 kB at 1280 px on the three uploads). Nothing is drawn into them: display labels
renumber after corrections, so overlays are drawn at render time (render.py), and the pixels depend only on the
video. The server passes runs/<video>/coaching/frames/ as the cache (store.frame_cache_path uses the same
f<frame>-w<width>.jpg names), so a frame survives corrections, regenerations and other runs of the same video. Files
are written atomically (uuid temp file + os.replace with retries against Windows reader locks, as corrections._write).
"""
import os
import time
import uuid
from pathlib import Path

import cv2

WIDTHS = (640, 1280)          # px: the report's half-width column and a full-width figure
JPEG_QUALITY = 85             # baseline JPEG; q85 at 1280 px is 128-186 kB median on the three uploads
GRAB_AHEAD = 45               # frames: decode forward up to this many frames instead of seeking (~ one seek's cost)
TIME_TOLERANCE = .5           # frames: the decoder's time must agree with the expected time within this much


class FrameError(Exception):
    """A frame that could not be read exactly."""


def frame_path(cache_dir, frame, width):
    """The cache file of a frame at a width: <cache_dir>/f<frame>-w<width>.jpg."""
    return Path(cache_dir)/f'f{int(frame)}-w{int(width)}.jpg'


def resize(image, width):
    """image scaled to width px (aspect kept, area interpolation); returned unchanged when it is not wider."""
    h, w = image.shape[:2]
    if w <= width:
        return image
    height = max(1, int(round(h*width/float(w))))
    return cv2.resize(image, (int(width), height), interpolation=cv2.INTER_AREA)


def encode_jpeg(image, width=None, quality=JPEG_QUALITY):
    """Baseline JPEG bytes of a BGR image, resized to width first when given."""
    if width is not None:
        image = resize(image, width)
    ok, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, int(quality), cv2.IMWRITE_JPEG_PROGRESSIVE, 0])
    if not ok:
        raise FrameError('JPEG encoding failed')
    return data.tobytes()


def iter_frames(video_path, frames, times=None, fps=None):
    """Decode the given absolute frame indices from the video in increasing order (each once), yielding
    (frame, bgr image or None, error text or None). times: {frame: expected absolute seconds} (frames[i].time of
    the result); without it a frame is expected at frame / fps. fps: the result's video.fps, else the capture's."""
    wanted = sorted({int(f) for f in frames})
    times = times or {}
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            for f in wanted:
                yield f, None, 'the video could not be opened'
            return
        rate = float(fps) if fps else float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not rate or rate <= 0:
            for f in wanted:
                yield f, None, 'the video has no frame rate'
            return
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        position = None                   # index of the frame the next read() returns
        for f in wanted:
            if f < 0 or (count and f >= count):
                yield f, None, 'frame index outside the video'
                continue
            if position is None or not (position <= f <= position+GRAB_AHEAD):
                capture.set(cv2.CAP_PROP_POS_FRAMES, f)
                position = f
            ok = True
            while ok and position < f:
                ok = capture.grab()
                position += 1
            ok, image = capture.read() if ok else (False, None)
            if not ok or image is None:
                position = None
                yield f, None, 'the decoder returned no picture'
                continue
            position = f+1
            stamp = float(capture.get(cv2.CAP_PROP_POS_MSEC))/1000.0
            if stamp <= 0 and f:
                stamp = f/rate             # pipeline.py: some decoders report 0 ms; the index is the time then
            expected = float(times[f]) if f in times else f/rate
            if abs(stamp-expected) > TIME_TOLERANCE/rate:
                yield f, None, f'decoded time {stamp:.4f} s is not the expected {expected:.4f} s'
                continue
            yield f, image, None
    finally:
        capture.release()


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    temp.write_bytes(data)
    for attempt in range(20):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 19:
                temp.unlink(missing_ok=True)
                raise
            time.sleep(.1)


def extract(video_path, frames, cache_dir=None, widths=(1280,), times=None, fps=None, quality=JPEG_QUALITY):
    """JPEGs of the given frames at the given widths. Cached files are reused without decoding; the others are
    decoded in one sorted pass. Returns {'jpegs': {(frame, width): bytes}, 'files': {(frame, width): path}
    (with a cache), 'errors': {frame: text}, 'decoded': n, 'cached': n, 'seconds': s}."""
    widths = tuple(int(w) for w in widths)
    for w in widths:
        if w not in WIDTHS:
            raise ValueError(f'width must be one of {WIDTHS}')
    start = time.perf_counter()
    out = {'jpegs': {}, 'files': {}, 'errors': {}, 'decoded': 0, 'cached': 0}
    todo = set()
    for f in {int(f) for f in frames}:
        for w in widths:
            path = frame_path(cache_dir, f, w) if cache_dir is not None else None
            if path is not None and path.is_file():
                out['jpegs'][(f, w)] = path.read_bytes()
                out['files'][(f, w)] = path
                out['cached'] += 1
            else:
                todo.add(f)
    if todo:
        for f, image, error in iter_frames(video_path, todo, times, fps):
            if image is None:
                out['errors'][f] = error
                continue
            out['decoded'] += 1
            for w in widths:
                if (f, w) in out['jpegs']:
                    continue
                data = encode_jpeg(image, w, quality)
                out['jpegs'][(f, w)] = data
                if cache_dir is not None:
                    path = frame_path(cache_dir, f, w)
                    _write(path, data)
                    out['files'][(f, w)] = path
    out['seconds'] = round(time.perf_counter()-start, 3)
    return out


class FrameProvider:
    """The PDF's frame source: provider(frame, width) -> JPEG bytes or None. prefetch(frames, widths) reads many
    frames in one sorted pass first (pdf.py calls it with every screenshot of the report). With a cache_dir the
    JPEGs are files shared across runs of the video; without one they are kept in memory."""

    def __init__(self, video_path, cache_dir=None, times=None, fps=None, quality=JPEG_QUALITY):
        self.video_path = Path(video_path)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.times = dict(times or {})
        self.fps = fps
        self.quality = quality
        self.memory = {}
        self.errors = {}
        self.decoded = 0

    def prefetch(self, frames, widths=(WIDTHS[0],)):
        got = extract(self.video_path, frames, self.cache_dir, widths, self.times, self.fps, self.quality)
        self.memory.update(got['jpegs'])
        self.errors.update(got['errors'])
        self.decoded += got['decoded']
        return got

    def __call__(self, frame, width=WIDTHS[0]):
        key = (int(frame), int(width))
        if key not in self.memory and key[0] not in self.errors:
            self.prefetch([key[0]], [key[1]])
        return self.memory.get(key)
