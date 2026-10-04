"""Local CUDA prototype. No uploads, API calls, roster guesses or off-screen paths."""
import os
import json
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
(ROOT / 'settings' / 'Ultralytics').mkdir(parents=True, exist_ok=True)
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / 'settings')
os.environ['YOLO_AUTOINSTALL'] = 'false'

import cv2
import numpy as np
import torch
from ultralytics import YOLO, settings
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from torchvision.ops import nms
from download_model import get_model
from vision import Appearance, FieldRegion, Ball, assign_teams, shirt

settings.update({'sync': False})
ROLES = {1: 'goalkeeper', 2: 'player', 3: 'referee'}


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


def ball_tiles(model, frame):
    """Magnify the full image in overlapping regions; don't assume ball is centered."""
    height, width = frame.shape[:2]
    size = min(800, width, height)
    def starts(length):
        return sorted(set(list(range(0, max(1, length-size), round(size*.8)))+[length-size]))
    regions = [(x, y) for y in starts(height) for x in starts(width)]
    output = []
    # Batch two tiles to bound VRAM use on a 16 GB card.
    for offset in range(0, len(regions), 2):
        batch = regions[offset:offset+2]
        predictions = model.predict([frame[y:y+size, x:x+size] for x, y in batch],
                                    imgsz=640, conf=.09, iou=.4, classes=[0], device=0, half=True, verbose=False)
        for (x, y), prediction in zip(batch, predictions):
            rows = prediction.boxes.data.cpu().numpy().copy()
            if len(rows):
                rows[:, :4] += [x, y, x, y]
                output.extend(rows.tolist())
    return output


def detect(model, frame, detail):
    # This checkpoint loses larger foreground players at high input sizes.
    # Keep the 960 whole-frame view and merge an additional detail pass.
    passes = []
    for size in sorted({960, detail}):
        result = model.predict(frame, imgsz=size, conf=.10, iou=.45,
                               classes=None, device=0, half=True, verbose=False)[0]
        passes.append(result.boxes.data.cpu())
    rows = torch.cat(passes)
    if not len(rows):
        return rows.numpy()
    # Person roles may disagree across scales; suppress those duplicates jointly.
    categories = (rows[:, 5] == 0).to(rows.dtype)
    separated = rows[:, :4] + categories[:, None]*max(frame.shape[:2])*2
    keep = nms(separated, rows[:, 4], .45)
    return rows[keep].numpy()


def preview(frame, observations, path):
    image = frame.copy()
    h, w = image.shape[:2]
    for person in observations['people']:
        x, y, bw, bh = person['box']
        color = (80, 210, 160) if person['role'] == 'player' else (70, 190, 255)
        a, b = (int(x*w), int(y*h)), (int((x+bw)*w), int((y+bh)*h))
        cv2.rectangle(image, a, b, color, 2)
        cv2.putText(image, f"{person['id']} {person['role']}", (a[0], max(16, a[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
    if observations['ball']:
        x, y, bw, bh = observations['ball']['box']
        cv2.circle(image, (int((x+bw/2)*w), int((y+bh/2)*h)), max(7, int(bw*w)), (255, 230, 70), 2)
    cv2.putText(image, f"{observations['time']:.2f}s | local track IDs; ball provisional", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
    encoded = cv2.imencode('.jpg', cv2.resize(image, (960, round(h/w*960))))[1]
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(encoded.tobytes())
    temporary.replace(path)


def run(video_path, output, config, progress, cancelled):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    gpu = gpu_info()
    meta = metadata(video_path)
    width, height, fps = meta['width'], meta['height'], meta['fps']
    start = float(config.get('start', 0))
    finish = min(meta['duration'], start+float(config.get('seconds', 60)))
    model_path = get_model()
    progress({'stage': 'Loading soccer detector and player appearance model', 'progress': 0, 'gpu': gpu})
    model = YOLO(str(model_path))
    if dict(model.names) != {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}:
        raise RuntimeError(f'Unexpected detector classes: {model.names}')
    appearance = Appearance()
    args = SimpleNamespace(track_high_thresh=.30, track_low_thresh=.10, new_track_thresh=.40,
                           track_buffer=60, match_thresh=.75, fuse_score=True,
                           gmc_method='sparseOptFlow', proximity_thresh=.3, appearance_thresh=.85,
                           with_reid=False, model='auto')
    tracker = BOTSORT(args, frame_rate=round(fps))
    tracker.args.with_reid = True
    tracker.encoder = appearance
    region = FieldRegion(config.get('boundaries', []))
    ball = Ball()
    capture = cv2.VideoCapture(str(video_path))
    first_frame = int(round(start*fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    frames, previous_boxes = [], []
    role_votes = {}
    segment = 0
    previous_thumbnail = None
    torch.cuda.reset_peak_memory_stats()
    (output/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    index, last_notice = first_frame, -10
    try:
        while not cancelled():
            ok, frame = capture.read()
            if not ok:
                break
            stamp = float(capture.get(cv2.CAP_PROP_POS_MSEC))/1000
            if stamp <= 0 and index:
                stamp = index/fps
            if stamp >= finish:
                break
            thumbnail = cv2.resize(frame, (160, 90))
            warp, reliable = region.update(frame, stamp, previous_boxes)
            if previous_thumbnail is not None and not reliable and np.mean(cv2.absdiff(thumbnail, previous_thumbnail)) > 65:
                tracker.reset()
                segment += 1
                ball = Ball()
            previous_thumbnail = thumbnail
            rows = detect(model, frame, int(config.get('imageSize', 1280)))
            humans = rows[rows[:, 5] != 0].copy()
            excluded = [normalized(row[:4], width, height) for row in humans if not region.contains(row[:4])]
            humans = np.array([row for row in humans if region.contains(row[:4])], np.float32).reshape(-1, 6)
            tracked = tracker.update(Boxes(humans, (height, width)), frame)
            people = []
            for row in tracked:
                box, local_id, score, role_id = row[:4], int(row[4]), float(row[5]), int(row[6])
                tid = f'T{segment*100000+local_id}'
                votes = role_votes.setdefault(tid, Counter())
                votes[ROLES.get(role_id, 'person')] += score
                feature = shirt(frame, box)
                people.append({'id': tid, 'box': normalized(box, width, height), 'score': round(score, 3),
                               'role': votes.most_common(1)[0][0], 'team': '?', 'evidence': 'observed',
                               'shirt': feature.tolist() if feature is not None else None})
            ball_rows = rows[rows[:, 5] == 0].tolist()
            if config.get('ballTiles', False) and (index-first_frame) % 2 == 0:
                ball_rows += ball_tiles(model, frame)
            if ball_rows:
                raw = torch.tensor(ball_rows, dtype=torch.float32)
                ball_rows = raw[nms(raw[:, :4], raw[:, 4], .3)].numpy().tolist()
            candidates = []
            for x1, y1, x2, y2, score, _ in ball_rows:
                if max(x2-x1, y2-y1) > max(width, height)*.035:
                    continue
                # Ball can be airborne: do not test its box against a ground polygon.
                candidates.append({'box': normalized([x1, y1, x2, y2], width, height),
                                   'score': round(score, 3), 'pixels': [x1, y1, x2, y2]})
            chosen = ball.update(candidates, stamp, warp, reliable, width)
            observation = {'time': round(stamp, 6), 'frame': index, 'people': people, 'ball': chosen,
                           'ballCandidates': [{k: v for k, v in c.items() if k != 'pixels'} for c in candidates],
                           'excluded': excluded, 'cameraReliable': reliable,
                           'boundary': (region.polygon/[width, height]).round(6).tolist() if region.polygon is not None else None}
            frames.append(observation)
            previous_boxes = [person['box'] for person in people]
            elapsed = time.perf_counter()-started
            if elapsed-last_notice >= 1:
                preview(frame, observation, output/'preview.jpg')
                progress({'stage': 'Tracking visible players and ball candidates', 'progress': (stamp-start)/(finish-start),
                          'videoTime': stamp, 'elapsed': round(elapsed, 1), 'frames': len(frames),
                          'processingFPS': round(len(frames)/elapsed, 1), 'people': len(people), 'warning': region.warning})
                last_notice = elapsed
            index += 1
    finally:
        capture.release()
    if not frames:
        raise RuntimeError('No video frames were processed.')
    progress({'stage': 'Aggregating team colors over player tracks'})
    teams = assign_teams(frames)
    for frame in frames:
        for person in frame['people']:
            person.pop('shirt', None)
    tracks = {}
    for frame in frames:
        for person in frame['people']:
            info = tracks.setdefault(person['id'], {'firstSeen': frame['time'], 'lastSeen': frame['time'], 'observations': 0})
            info.update(lastSeen=frame['time'], role=person['role'], team=person['team'])
            info['observations'] += 1
    elapsed = time.perf_counter()-started
    report = {'version': 1, 'video': meta, 'gpu': gpu, 'config': config, 'frames': frames, 'tracks': tracks,
              'teams': teams, 'status': 'cancelled' if cancelled() else 'complete',
              'summary': {'processedFrames': len(frames), 'trackSegments': len(tracks),
                          'meanVisiblePeople': round(sum(len(f['people']) for f in frames)/len(frames), 1),
                          'ballCandidateFrameFraction': round(sum(f['ball'] is not None for f in frames)/len(frames), 3),
                          'elapsedSeconds': round(elapsed, 2), 'processingFPS': round(len(frames)/elapsed, 2),
                          'peakGPUMemoryGB': round(torch.cuda.max_memory_allocated()/1024**3, 2)},
              'limitations': ['Local track IDs are not jersey numbers or verified identities.',
                              'No long-gap identity recovery or metric pitch calibration in this first prototype.',
                              'Off-screen positions are unknown; no missing observations are fabricated.',
                              'Ball observations are provisional candidates, not verified ball accuracy.',
                              'Team colors are inferred; role labels may be mistaken.']}
    (output/'result.json').write_text(json.dumps(report, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    progress({'stage': report['status'], 'progress': 1, 'summary': report['summary'], 'teams': teams})
    return report
