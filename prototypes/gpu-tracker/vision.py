"""OSNet appearance descriptors and provisional ball selection."""
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort


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

    def embed(self, frame, boxes):
        """L2-normalized 512-d descriptors for [x1, y1, x2, y2] pixel boxes."""
        h, w = frame.shape[:2]
        crops = []
        for x1, y1, x2, y2 in boxes:
            x1, y1 = min(w-1, max(0, int(x1))), min(h-1, max(0, int(y1)))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
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


class Ball:
    def __init__(self):
        self.point = None
        self.last = -100
        self.support = 0

    def update(self, candidates, timestamp, motion, reliable, width):
        """motion: 3x3 pixel transform from the previous frame (camera compensation)."""
        if self.point is not None and reliable:
            p = motion @ np.array([self.point[0], self.point[1], 1.0])
            self.point = p[:2]/p[2]
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
