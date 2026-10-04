"""Small local helpers. Team labels and track IDs are estimates, not roster IDs."""
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from sklearn.cluster import KMeans


class Appearance:
    """Existing OSNet x0.25 descriptor, on CPU; detector runs on CUDA."""
    def __init__(self):
        path = Path(__file__).resolve().parents[2] / 'public/models/osnet-x025.onnx'
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        self.input = self.session.get_inputs()[0].name
        self.mean = np.array([.485, .456, .406], np.float32)
        self.std = np.array([.229, .224, .225], np.float32)

    def __call__(self, frame, boxes):
        h, w = frame.shape[:2]
        crops = []
        for cx, cy, bw, bh, *_ in boxes:
            x1, y1 = min(w-1, max(0, int(cx-bw/2))), min(h-1, max(0, int(cy-bh/2)))
            x2, y2 = min(w, int(cx+bw/2)), min(h, int(cy+bh/2))
            crop = frame[y1:max(y1+1, y2), x1:max(x1+1, x2)]
            crop = cv2.cvtColor(cv2.resize(crop, (128, 256)), cv2.COLOR_BGR2RGB)
            crops.append(((crop.astype(np.float32)/255-self.mean)/self.std).transpose(2, 0, 1))
        result = []
        for offset in range(0, len(crops), 16):
            batch = np.zeros((16, 3, 256, 128), np.float32)
            count = min(16, len(crops)-offset)
            batch[:count] = crops[offset:offset+count]
            features = self.session.run(None, {self.input: batch})[0][:count]
            features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-9)
            result.extend(features)
        return result


def shirt(frame, box):
    x1, y1, x2, y2 = box
    w, h = x2-x1, y2-y1
    patch = frame[max(0, int(y1+h*.2)):int(y1+h*.5), max(0, int(x1+w*.28)):int(x1+w*.72)]
    if patch.size < 60:
        return None
    hsv = cv2.cvtColor(cv2.resize(patch, (12, 12)), cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    hue, sat, val = hsv[:, 0]/180*12, hsv[:, 1]/255, hsv[:, 2]/255
    dark = np.clip((.46-val)/.24, 0, 1)
    neutral = (1-dark)*np.clip((.3-sat)/.22, 0, 1)
    chroma = 1-dark-neutral
    hist = np.zeros(14)
    lower = hue.astype(int) % 12
    fraction = hue-hue.astype(int)
    np.add.at(hist, lower, chroma*(1-fraction))
    np.add.at(hist, (lower+1) % 12, chroma*fraction)
    hist[12], hist[13] = dark.sum(), neutral.sum()
    return hist/hist.sum()


def assign_teams(frames):
    """Vote over each track, fit two kits using clear players, retain unknowns."""
    tracks = {}
    for frame in frames:
        for person in frame['people']:
            tracks.setdefault(person['id'], []).append(person)
    profiles = {}
    for tid, observations in tracks.items():
        samples = [p['shirt'] for p in observations if p.get('shirt') is not None
                   and p['score'] >= .4 and p['box'][3] >= .035 and p['role'] == 'player']
        if len(samples) >= 8:
            profiles[tid] = np.mean(samples, axis=0)
    if len(profiles) < 6:
        return {'learned': False, 'reason': 'Not enough clear player tracks to learn two kits.'}
    ids = list(profiles)
    features = np.sqrt(np.array([profiles[tid] for tid in ids]))
    # Limit long, large foreground tracks to one vote each.
    clustering = KMeans(n_clusters=2, random_state=42, n_init=10).fit(features)
    counts = np.bincount(clustering.labels_, minlength=2)
    centers = clustering.cluster_centers_
    if min(counts) < 3 or np.linalg.norm(centers[0]-centers[1]) < .45:
        return {'learned': False, 'reason': 'Two distinct, supported kit groups were not found.'}
    # Stable naming within the run only; this is not home/away or a jersey number.
    ordering = np.argsort([c[12] for c in centers])
    labels = {int(ordering[0]): 'A', int(ordering[1]): 'B'}
    for tid, observations in tracks.items():
        profile = profiles.get(tid)
        if profile is None:
            continue
        distances = np.linalg.norm(centers-np.sqrt(profile), axis=1)
        best = int(np.argmin(distances))
        if distances[best] > .7 or abs(distances[0]-distances[1]) < .18:
            continue
        for person in observations:
            if person['role'] == 'player':
                person['team'] = labels[best]
    return {'learned': True, 'support': counts.tolist(), 'meaning': 'Kit groups, not roster identities.'}


class FieldRegion:
    """Propagate optional user-drawn field boundaries with background motion.

    This is an image-space exclusion region, NOT metric field calibration.
    Weak flow invalidates the region until another user anchor is encountered.
    """
    def __init__(self, anchors):
        self.anchors = sorted(anchors, key=lambda a: a['time'])
        self.next_anchor = 0
        self.polygon = None
        self.previous = None
        self.last_time = None
        self.warning = None

    def update(self, frame, timestamp, previous_boxes):
        h, w = frame.shape[:2]
        scale = 960/w
        gray = cv2.cvtColor(cv2.resize(frame, (960, round(h*scale))), cv2.COLOR_BGR2GRAY)
        warp = np.eye(2, 3, dtype=np.float32)
        reliable = False
        if self.previous is not None:
            mask = np.full_like(self.previous, 255)
            mask[:round(gray.shape[0]*.25)] = 0
            for box in previous_boxes:
                x, y, bw, bh = box
                a = (int(x*w*scale), int(y*h*scale))
                b = (int((x+bw)*w*scale), int((y+bh)*h*scale))
                cv2.rectangle(mask, a, b, 0, -1)
            points = cv2.goodFeaturesToTrack(self.previous, 600, .01, 8, mask=mask)
            if points is not None and len(points) >= 12:
                moved, status, error = cv2.calcOpticalFlowPyrLK(self.previous, gray, points, None)
                if moved is not None:
                    keep = status.ravel().astype(bool) & (error.ravel() < 30)
                    if keep.sum() >= 12:
                        matrix, inliers = cv2.estimateAffinePartial2D(points[keep], moved[keep], method=cv2.RANSAC, ransacReprojThreshold=2)
                        if matrix is not None and inliers is not None:
                            zoom = np.sqrt(abs(np.linalg.det(matrix[:, :2])))
                            reliable = bool(inliers.mean() > .6 and .85 < zoom < 1.18 and np.linalg.norm(matrix[:, 2]) < 180)
                            if reliable:
                                warp = matrix.astype(np.float32)
                                warp[:, 2] /= scale
            if self.polygon is not None:
                if reliable:
                    self.polygon = cv2.transform(self.polygon[None].astype(np.float32), warp)[0]
                else:
                    self.polygon = None
                    self.warning = 'Field boundary lost; add another boundary keyframe. Detections are retained.'
        while self.next_anchor < len(self.anchors) and self.anchors[self.next_anchor]['time'] <= timestamp+1e-4:
            anchor = self.anchors[self.next_anchor]
            # Do not transplant a stale screen polygon to a later starting view.
            if timestamp-anchor['time'] < .12:
                self.polygon = np.array(anchor['points'], np.float32)*[w, h]
                self.warning = None
            self.next_anchor += 1
        self.previous, self.last_time = gray, timestamp
        return warp, reliable

    def contains(self, box):
        if self.polygon is None:
            return True
        x1, y1, x2, y2 = box
        return cv2.pointPolygonTest(self.polygon.astype(np.float32), (float((x1+x2)/2), float(y2)), True) >= -8


class Ball:
    def __init__(self):
        self.point = None
        self.last = -100
        self.support = 0

    def update(self, candidates, timestamp, warp, reliable, width):
        if self.point is not None and reliable:
            self.point = cv2.transform(self.point[None, None].astype(np.float32), warp)[0, 0]
        if timestamp-self.last > .4 or not reliable:
            self.point = None
            self.support = 0
        ranked = []
        for candidate in candidates:
            box = candidate['pixels']
            point = np.array([(box[0]+box[2])/2, (box[1]+box[3])/2])
            displacement = np.linalg.norm(point-self.point) if self.point is not None else 0
            if self.point is not None and displacement > width*.09:
                continue
            ranked.append((candidate['score']-displacement/(width*.18), candidate, point))
        ranked.sort(key=lambda r: -r[0])
        if not ranked or len(ranked) > 1 and ranked[0][0]-ranked[1][0] < .12:
            return None
        _, candidate, self.point = ranked[0]
        self.last = timestamp
        self.support += 1
        if self.support < 2 or candidate['score'] < .12:
            return None
        return {k: v for k, v in candidate.items() if k != 'pixels'} | {'evidence': 'observed', 'identity': 'ball candidate'}
