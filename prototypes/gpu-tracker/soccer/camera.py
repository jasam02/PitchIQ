"""Camera-motion compensation.

Frame-to-frame motion is estimated from background features (people and the top overlay band are
masked) with sparse optical flow and a robust similarity fit (pan, zoom, small rotation). It is used
to: detect camera cuts, express foot points in camera-stabilized coordinates (so a pan is not read as
player movement), carry user field boundaries and the pitch calibration through pans and zooms.
When motion is unreliable, nothing is extrapolated: stabilized positions accumulate drift (which
weakens spatial evidence), boundaries and calibration are dropped until the next keyframe.
"""
import cv2
import numpy as np

from . import calibration
from .geometry import foot_point, transform_points

WORK_W = 960


class CameraMotion:
    def __init__(self):
        self.previous = None
        self.thumbnail = None

    def reset(self):
        self.previous = self.thumbnail = None

    def estimate(self, frame, person_boxes):
        """person_boxes: normalized [x, y, w, h] boxes to mask. Returns (M, reliable, cut) where M is a
        3x3 pixel transform from the previous frame to this one."""
        h, w = frame.shape[:2]
        scale = WORK_W/w
        gray = cv2.cvtColor(cv2.resize(frame, (WORK_W, round(h*scale))), cv2.COLOR_BGR2GRAY)
        thumbnail = cv2.resize(frame, (160, 90))
        M, reliable = np.eye(3), False
        if self.previous is not None and self.previous.shape == gray.shape:
            mask = np.full_like(self.previous, 255)
            mask[:round(gray.shape[0]*.25)] = 0  # scoreboard / broadcast overlays do not move with the camera
            for x, y, bw, bh in person_boxes:
                cv2.rectangle(mask, (int(x*WORK_W), int(y*gray.shape[0])), (int((x+bw)*WORK_W), int((y+bh)*gray.shape[0])), 0, -1)
            points = cv2.goodFeaturesToTrack(self.previous, 600, .01, 8, mask=mask)
            if points is not None and len(points) >= 12:
                moved, status, error = cv2.calcOpticalFlowPyrLK(self.previous, gray, points, None)
                if moved is not None:
                    keep = status.ravel().astype(bool) & (error.ravel() < 30)
                    if keep.sum() >= 12:
                        matrix, inliers = cv2.estimateAffinePartial2D(points[keep], moved[keep], method=cv2.RANSAC, ransacReprojThreshold=2)
                        if matrix is not None and inliers is not None:
                            zoom = float(np.sqrt(abs(np.linalg.det(matrix[:, :2]))))
                            reliable = bool(inliers.mean() > .6 and .85 < zoom < 1.18 and np.linalg.norm(matrix[:, 2]) < 180)
                            if reliable:
                                M = np.vstack([matrix, [0, 0, 1]]).astype(np.float64)
                                M[:2, 2] /= scale
        cut = bool(self.thumbnail is not None and not reliable and float(np.mean(cv2.absdiff(thumbnail, self.thumbnail))) > 65)
        self.previous, self.thumbnail = gray, thumbnail
        return M, reliable, cut


class CameraState:
    """Camera segment (changes on every cut) and the transform from the current frame to the
    segment's first frame. Stabilized coordinates are comparable only within one segment."""

    def __init__(self, width, height):
        self.width, self.height = width, height
        self.segment = 0
        self.T = np.eye(3)
        self.drift = 0
        self.started = False

    def update(self, M, reliable, cut):
        if not self.started:
            self.started = True
            return
        if cut:
            self.segment += 1
            self.T = np.eye(3)
            self.drift = 0
        elif reliable:
            self.T = self.T @ np.linalg.inv(M)
        else:
            self.drift += 1

    def stabilize_point(self, x, y):
        """A normalized image point in normalized coordinates of the segment's first frame."""
        (sx, sy), = transform_points(self.T, [(x*self.width, y*self.height)])
        return sx/self.width, sy/self.height

    def stabilize(self, box):
        """Foot point and box height in normalized coordinates of the segment's first frame."""
        fx, fy = foot_point(box)
        (x, y), = transform_points(self.T, [(fx*self.width, fy*self.height)])
        s = float(np.sqrt(abs(np.linalg.det(self.T[:2, :2]))))
        return x/self.width, y/self.height, box[3]*s


class Keyframes:
    """User boundary polygons and pitch calibrations, applied at their keyframe time and carried
    through reliable camera motion. Unreliable motion drops them (with a warning) until the next one."""

    def __init__(self, boundaries, calibrations, width, height):
        self.width, self.height = width, height
        self.boundaries = sorted(boundaries or [], key=lambda a: a['time'])
        self.calibrations = sorted(calibrations or [], key=lambda a: a['time'])
        self.next_boundary = self.next_calibration = 0
        self.polygon = None   # pixels
        self.H = None         # image pixels -> pitch metres
        self.rms = None
        self.warning = None

    def update(self, timestamp, M, reliable):
        if self.polygon is not None:
            if reliable:
                self.polygon = transform_points(M, self.polygon)
            else:
                self.polygon = None
                self.warning = 'Field boundary lost on an unreliable camera move; add another boundary keyframe.'
        if self.H is not None:
            if reliable:
                self.H = self.H @ np.linalg.inv(M)
            else:
                self.H = None
                self.warning = 'Pitch calibration lost on an unreliable camera move; add another calibration keyframe.'
        while self.next_boundary < len(self.boundaries) and self.boundaries[self.next_boundary]['time'] <= timestamp+1e-4:
            anchor = self.boundaries[self.next_boundary]
            # Do not transplant a stale screen polygon to a later starting view.
            if timestamp-anchor['time'] < .12:
                self.polygon = np.asarray(anchor['points'], np.float64)*[self.width, self.height]
                self.warning = None
            self.next_boundary += 1
        while self.next_calibration < len(self.calibrations) and self.calibrations[self.next_calibration]['time'] <= timestamp+1e-4:
            anchor = self.calibrations[self.next_calibration]
            if timestamp-anchor['time'] < .12:
                try:
                    self.H, self.rms = calibration.solve(anchor['points'], self.width, self.height)
                    self.warning = None
                except ValueError as error:
                    self.H, self.warning = None, f'Calibration at {anchor["time"]:.2f}s ignored: {error}'
            self.next_calibration += 1

    def boundary(self):
        """User polygon in normalized coordinates, or None."""
        if self.polygon is None:
            return None
        return [(float(x)/self.width, float(y)/self.height) for x, y in self.polygon]

    def to_pitch(self, box):
        if self.H is None:
            return None
        fx, fy = foot_point(box)
        return calibration.to_pitch(self.H, (fx*self.width, fy*self.height))

    def to_pitch_point(self, point):
        if self.H is None:
            return None
        return calibration.to_pitch(self.H, (point[0]*self.width, point[1]*self.height))

    def pitch_outline(self):
        return calibration.pitch_outline(self.H, self.width, self.height) if self.H is not None else None

    def normalized_motion(self, M):
        """M expressed in normalized image coordinates (for carrying pitch boundary lines)."""
        S = np.diag([self.width, self.height, 1.0])
        return np.linalg.inv(S) @ M @ S
