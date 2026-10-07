"""Local CUDA prototype. No uploads, API calls, roster guesses or off-screen paths.

Detection (soccer YOLOv8x on CUDA) feeds the soccer tracking layer in soccer/: pitch detection,
foot-point field filtering, BoT-SORT local tracks, OSNet + kit appearance, team/role votes and the
global identity manager that keeps one identity per real person across exits and returns.
"""
import os
import json
import time
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
from soccer.identity import long_id
from soccer.local import BotSortTracker
from soccer.tracker import SoccerTracker, config_from
from vision import Appearance, Ball

settings.update({'sync': False})
COLORS = {'A': (255, 190, 90), 'B': (90, 170, 250), 'GOALKEEPER': (220, 120, 230), 'REFEREE': (40, 200, 255), 'UNCERTAIN': (170, 170, 170)}


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


def _colour(person):
    role = person['role']
    if person['id'] is None:
        return (255, 255, 255) if role == 'CANDIDATE' else COLORS['UNCERTAIN']
    if role.startswith('GOALKEEPER'):
        return COLORS['GOALKEEPER']
    if role == 'REFEREE':
        return COLORS['REFEREE']
    return COLORS.get(person['team'], COLORS['UNCERTAIN'])


def preview(frame, observation, path):
    image = frame.copy()
    h, w = image.shape[:2]
    polygon = (observation.get('pitch') or {}).get('polygon')
    if polygon:
        cv2.polylines(image, [np.int32(np.asarray(polygon)*[w, h])], True, (110, 220, 160), 2)
    for rejected in observation['rejected']:
        x, y, bw, bh = rejected['box']
        cv2.rectangle(image, (int(x*w), int(y*h)), (int((x+bw)*w), int((y+bh)*h)), (80, 80, 235), 1)
    for person in observation['people']:
        x, y, bw, bh = person['box']
        color = _colour(person)
        a, b = (int(x*w), int(y*h)), (int((x+bw)*w), int((y+bh)*h))
        cv2.rectangle(image, a, b, color, 2)
        text = f"{person['label']} T{person['track']} {person['role'].replace('_TEAM_', ' ')} {round(person['identity']*100)}%"
        cv2.putText(image, text, (a[0], max(16, a[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
    if observation['ball']:
        x, y, bw, bh = observation['ball']['box']
        cv2.circle(image, (int((x+bw/2)*w), int((y+bh/2)*h)), max(7, int(bw*w)), (255, 230, 70), 2)
    cv2.putText(image, f"{observation['time']:.2f}s | global ID, local track, role, identity confidence", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
    encoded = cv2.imencode('.jpg', cv2.resize(image, (960, round(h/w*960))))[1]
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(encoded.tobytes())
    temporary.replace(path)


def run(video_path, output, config, progress, cancelled, previous=None):
    """previous: saved identities of an earlier run of this video, to continue its global identities."""
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
    tracker = SoccerTracker(width, height, config_from(config), BotSortTracker(fps), appearance.embed, previous)
    ball = Ball()
    capture = cv2.VideoCapture(str(video_path))
    first_frame = int(round(start*fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    frames = []
    torch.cuda.reset_peak_memory_stats()
    (output/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    log = (output/'events.log').open('w', encoding='utf-8')
    index, last_notice, logged = first_frame, -10, 0
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
            rows = detect(model, frame, int(config.get('imageSize', 1280)))
            observation, motion, reliable = tracker.step(frame, stamp, rows[rows[:, 5] != 0])
            if observation['cut']:
                ball = Ball()
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
            chosen = ball.update(candidates, stamp, motion, reliable, width)
            observation = {'time': round(stamp, 6), 'frame': index, **observation, 'ball': chosen,
                           'ballCandidates': [{k: v for k, v in c.items() if k != 'pixels'} for c in candidates]}
            frames.append(observation)
            # Every identity decision is logged; accepted re-identifications also go to the console.
            for event in tracker.events[logged:]:
                log.write(f"[{event['time']:.2f}s] {event['kind'].upper()}\n{event['message']}\n\n")
                if event['kind'] in ('reid', 'swap-corrected', 'sanity'):
                    print(f"[{event['time']:.2f}s] {event['message']}", flush=True)
            logged = len(tracker.events)
            elapsed = time.perf_counter()-started
            if elapsed-last_notice >= 1:
                preview(frame, observation, output/'preview.jpg')
                counts = tracker.ids.summary()
                progress({'stage': 'Tracking match participants and global identities', 'progress': (stamp-start)/(finish-start),
                          'videoTime': stamp, 'elapsed': round(elapsed, 1), 'frames': len(frames),
                          'processingFPS': round(len(frames)/elapsed, 1), 'people': len(observation['people']),
                          'identities': {k: len(v) for k, v in counts.items()}, 'warning': tracker.keyframes.warning})
                last_notice = elapsed
            index += 1
    finally:
        capture.release()
        log.close()
    if not frames:
        raise RuntimeError('No video frames were processed.')
    progress({'stage': 'Writing global identities and tracking data'})
    report = build_report(frames, tracker, meta, gpu, config, time.perf_counter()-started, cancelled(), previous)
    snapshot = tracker.snapshot() | {'run': output.name, 'start': start, 'end': frames[-1]['time']}
    (output/'identities.json').write_text(json.dumps(snapshot, separators=(',', ':')), encoding='utf-8')
    (output/'result.json').write_text(json.dumps(report, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    progress({'stage': report['status'], 'progress': 1, 'summary': report['summary'], 'teams': report['teams']})
    return report


def build_report(frames, tracker, meta, gpu, config, elapsed, cancelled, previous):
    tracks, people, identified = {}, 0, 0
    for frame in frames:
        for person in frame['people']:
            info = tracks.setdefault(person['track'], {'firstSeen': frame['time'], 'lastSeen': frame['time'], 'observations': 0, 'ids': {}})
            info['lastSeen'] = frame['time']
            info['observations'] += 1
            if person['id']:
                info['ids'][person['id']] = info['ids'].get(person['id'], 0)+1
            if person['state'] not in ('rejected', 'unknown'):
                people += 1
                identified += person['id'] is not None
    match = tracker.ids.summary()
    events = tracker.events
    kinds = {}
    for event in events:
        kinds[event['kind']] = kinds.get(event['kind'], 0)+1
    count = lambda team: sum(1 for p in match['players'] if p['team'] == team)
    summary = {'processedFrames': len(frames), 'localTrackSegments': len(tracks),
               'globalIdentities': {'teamA': count('A'), 'teamB': count('B'), 'goalkeepers': len(match['goalkeepers']), 'referees': len(match['referees'])},
               'reidentifications': kinds.get('reid', 0), 'deferredDecisions': kinds.get('deferred', 0), 'newIdentities': kinds.get('new-identity', 0),
               'swapCorrections': kinds.get('swap-corrected', 0), 'crossingWarnings': kinds.get('swap-uncertain', 0), 'sanityWarnings': kinds.get('sanity', 0),
               'identifiedShare': round(identified/people, 3) if people else 0.0, 'rejectedDetections': tracker.rejected_counts,
               'cameraCuts': tracker.cuts, 'meanVisiblePeople': round(sum(len(f['people']) for f in frames)/len(frames), 1),
               'ballCandidateFrameFraction': round(sum(f['ball'] is not None for f in frames)/len(frames), 3),
               'elapsedSeconds': round(elapsed, 2), 'processingFPS': round(len(frames)/elapsed, 2),
               'peakGPUMemoryGB': round(torch.cuda.max_memory_allocated()/1024**3, 2)}
    return {'version': 2, 'video': meta, 'gpu': gpu, 'config': config, 'frames': frames, 'match': match, 'tracks': tracks,
            'events': events, 'issues': tracker.issues[-500:], 'teams': tracker.model.to_json(),
            'continuedFrom': (previous or {}).get('run'), 'status': 'cancelled' if cancelled else 'complete', 'summary': summary,
            'labels': {p['id']: long_id(p['id']) for group in match.values() for p in group},
            'limitations': ['Global IDs (A-07, GK-1, REF-1) are tracking identities, not jersey numbers or names.',
                            'Identity is kept only when the evidence is clear; uncertain people stay unidentified rather than guessed.',
                            'OSNet is a general pedestrian model: same-kit teammates can look alike, so some returns stay unresolved.',
                            'Pitch coordinates exist only for frames carried from a calibration keyframe.',
                            'Off-screen positions are unknown; no missing observations are fabricated.',
                            'Ball observations are provisional candidates, not verified ball accuracy.']}
