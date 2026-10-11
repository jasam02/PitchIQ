"""Camera-motion compensation.

Frame-to-frame motion is estimated from background features (people and the top overlay band are
masked) with sparse optical flow and a robust similarity fit (pan, zoom, small rotation). It is used
to: detect camera cuts, express foot points in camera-stabilized coordinates (so a pan is not read as
player movement), carry user field boundaries and the pitch model's camera homography through pans
and zooms. When motion is unreliable, nothing is extrapolated for long: stabilized positions
accumulate drift (which weakens spatial evidence), boundaries are dropped until the next keyframe and
the pitch model's confidence falls quickly until it is let go (pitch_model.py).
"""
import cv2
import numpy as np

from . import calibration
from .geometry import foot_point, transform_points
from .pitch_model import LOW_LEVEL, PitchTracker

WORK_W = 960


def _sane_homography(Hm, w, h):
    """A frame-to-frame homography of a panning camera: nearly affine, orientation kept, corners
    moving less than a third of the frame."""
    if Hm is None or not np.isfinite(Hm).all() or abs(Hm[2, 2]) < 1e-9:
        return False
    Hm = Hm/Hm[2, 2]
    if abs(Hm[2, 0]) > 2e-4 or abs(Hm[2, 1]) > 2e-4 or np.linalg.det(Hm[:2, :2]) <= 0:
        return False
    corners = np.float64([[0, 0], [w, 0], [w, h], [0, h]])
    moved = transform_points(Hm, corners)
    return bool(np.hypot(*(moved-corners).T).max() < max(w, h)/3)


class CameraMotion:
    def __init__(self):
        self.previous = None
        self.thumbnail = None
        self.homography = None   # the latest reliable frame-to-frame homography (pixels), for carrying the pitch model

    def reset(self):
        self.previous = self.thumbnail = self.homography = None

    def estimate(self, frame, person_boxes):
        """person_boxes: normalized [x, y, w, h] boxes to mask. Returns (M, reliable, cut) where M is a
        3x3 pixel transform from the previous frame to this one."""
        h, w = frame.shape[:2]
        scale = WORK_W/w
        gray = cv2.cvtColor(cv2.resize(frame, (WORK_W, round(h*scale))), cv2.COLOR_BGR2GRAY)
        thumbnail = cv2.resize(frame, (160, 90))
        M, reliable = np.eye(3), False
        self.homography = None
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
                                # A panning camera moves the ground plane by a homography, which the similarity
                                # only approximates; composed over seconds of panning the approximation drifts,
                                # so the pitch model is carried by the full homography when enough background
                                # points support one.
                                if keep.sum() >= 40:
                                    Hm, mask = cv2.findHomography(points[keep], moved[keep], cv2.RANSAC, 3.0)
                                    if Hm is not None and mask is not None and mask.mean() > .6 and _sane_homography(Hm, WORK_W, gray.shape[0]):
                                        S = np.diag([scale, scale, 1.0])
                                        self.homography = np.linalg.inv(S) @ (Hm/Hm[2, 2]) @ S
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
    """User boundary polygons and pitch calibrations. Boundaries apply at their keyframe time and are
    carried through reliable camera motion (unreliable motion drops them, with a warning, until the
    next one). Calibrations anchor the tracked pitch model (`pitch`), which carries the camera
    homography itself; the pitch size is learned once from all calibration keyframes together."""

    def __init__(self, boundaries, calibrations, width, height, size=None):
        self.width, self.height = width, height
        self.boundaries = sorted(boundaries or [], key=lambda a: a['time'])
        self.calibrations = sorted(calibrations or [], key=lambda a: a['time'])
        self.next_boundary = self.next_calibration = 0
        self.polygon = None   # pixels
        self.size = size or calibration.learn_size(self.calibrations, width, height) or calibration.field.PitchSize()
        self.pitch = PitchTracker(width, height, self.size)
        self.rms = None
        self.warning = None
        self.last_time = None
        self.fresh = False    # a calibration keyframe anchored the model on this update

    @property
    def H(self):
        """Image pixels -> pitch metres while the pitch model is trusted (MEDIUM or better), else None."""
        if self.pitch.H is None or self.pitch.level == LOW_LEVEL:
            return None
        try:
            return np.linalg.inv(self.pitch.H)
        except np.linalg.LinAlgError:
            return None

    def update(self, timestamp, M, reliable, cut=False, homography=None):
        """homography: the frame-to-frame homography when the camera motion estimate found one; it
        carries the pitch model more faithfully than the similarity M through long pans."""
        dt = max(0.0, timestamp-self.last_time) if self.last_time is not None else 0.0
        self.last_time = timestamp
        self.fresh = False
        if self.polygon is not None:
            if reliable:
                self.polygon = transform_points(M, self.polygon)
            else:
                self.polygon = None
                self.warning = 'Field boundary lost on an unreliable camera move; add another boundary keyframe.'
        had = self.pitch.H is not None
        self.pitch.carry(homography if homography is not None and reliable else M, reliable, cut, dt, timestamp)
        if had and self.pitch.H is None:
            self.warning = f'Pitch calibration lost ({self.pitch.lost}); add another calibration keyframe.'
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
                    H, self.rms = calibration.solve(anchor['points'], self.width, self.height, size=self.size)
                    if self.pitch.anchor(H, timestamp, self.rms):
                        self.warning, self.fresh = None, True
                    else:
                        self.warning = f'Calibration at {anchor["time"]:.2f}s ignored: it does not describe a pitch plane in front of the camera'
                except ValueError as error:
                    self.warning = f'Calibration at {anchor["time"]:.2f}s ignored: {error}'
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
        return self.pitch.to_pitch((fx*self.width, fy*self.height))

    def to_pitch_point(self, point):
        if self.H is None:
            return None
        return self.pitch.to_pitch((point[0]*self.width, point[1]*self.height))

    def pitch_outline(self):
        return self.pitch.outline() if self.H is not None else None

    def normalized_motion(self, M):
        """M expressed in normalized image coordinates (for carrying pitch boundary lines)."""
        S = np.diag([self.width, self.height, 1.0])
        return np.linalg.inv(S) @ M @ S
