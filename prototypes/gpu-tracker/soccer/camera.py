"""Camera-motion compensation.

Frame-to-frame motion is estimated from background features (people and the top overlay band are
masked) with sparse optical flow and a robust similarity fit (pan, zoom, small rotation). It is used
to: detect camera cuts, express foot points in camera-stabilized coordinates (so a pan is not read as
player movement), carry user field boundaries and the pitch model's camera homography through pans
and zooms. When motion is unreliable, nothing is extrapolated for long: stabilized positions
accumulate drift (which weakens spatial evidence), boundaries are dropped until the next keyframe and
the pitch model's confidence falls quickly until it is let go (pitch_model.py).
"""
import math

import cv2
import numpy as np

from . import calibration
from .geometry import foot_point, transform_points
from .pitch_model import LOW_LEVEL, PitchTracker

WORK_W = 960
KEYFRAME_MAX_RMS = 4.0   # metres: landmarks disagreeing by more than this never anchor; below it the residual lowers the observability
REGISTER_FEATURES = 2500 # SIFT features kept per view for direct re-registration
REGISTER_RATIO = .75     # Lowe ratio for feature matches
REGISTER_MIN_MATCHES = 40    # inlier matches a direct registration needs
REGISTER_MAX_RMS = 3.0       # pixels (at WORK_W) of inlier residual it may have
REGISTER_MIN_OVERLAP = .3    # share of the current frame the keyframe's view must cover
REGISTER_VIEWS = 8           # keyframe views remembered
RELAY_VIEWS = 6              # relay views remembered (frames re-registered to a keyframe, kept when its view is leaving)
RELAY_BELOW = 120            # matches to a keyframe view below which the current frame becomes a relay
RELAY_OVERLAP = .6           # ... or when the keyframe's view covers less of the frame than this
COARSE_W = 320           # width of the frame used for the coarse (phase-correlation) pan estimate
COARSE_RESPONSE = .25    # phase-correlation peak response needed to trust a coarse pan estimate
ROUND_TRIP_PX = 1.5      # a flow correspondence must return to within this of its start when tracked back


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
    """Frame-to-frame motion. Feature flow (checked forward and backward) gives a full similarity and,
    with enough background points, a homography: quality 'flow'. When features are too few or
    inconsistent (a fast pan over flat grass, motion blur), a coarse whole-frame translation from
    phase correlation stands in: quality 'coarse', less precise, but it keeps carrying the pitch model
    and the local tracks through the pan instead of declaring a camera cut. A cut needs both to fail
    and the frame content to change abruptly."""

    def __init__(self):
        self.previous = None
        self.thumbnail = None
        self.coarse = None
        self.homography = None   # the latest reliable frame-to-frame homography (pixels), for carrying the pitch model
        self.quality = 'none'    # flow | coarse | none: how the last motion was measured

    def reset(self):
        self.previous = self.thumbnail = self.coarse = self.homography = None
        self.quality = 'none'

    def coarse_shift(self, gray_small):
        """Translation (pixels of the small frame) that explains the previous small frame, or None."""
        if self.coarse is None or self.coarse.shape != gray_small.shape:
            return None
        window = cv2.createHanningWindow(gray_small.shape[::-1], cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(self.coarse.astype(np.float32), gray_small.astype(np.float32), window)
        if not all(math.isfinite(v) for v in (dx, dy, response)) or response < COARSE_RESPONSE:
            return None
        if abs(dx) > gray_small.shape[1]*.4 or abs(dy) > gray_small.shape[0]*.4:
            return None
        return dx, dy

    def estimate(self, frame, person_boxes):
        """person_boxes: normalized [x, y, w, h] boxes to mask. Returns (M, reliable, cut) where M is a
        3x3 pixel transform from the previous frame to this one."""
        h, w = frame.shape[:2]
        scale = WORK_W/w
        gray = cv2.cvtColor(cv2.resize(frame, (WORK_W, round(h*scale))), cv2.COLOR_BGR2GRAY)
        thumbnail = cv2.resize(frame, (160, 90))
        small = cv2.resize(gray, (COARSE_W, max(2, round(gray.shape[0]*COARSE_W/WORK_W))), interpolation=cv2.INTER_AREA)
        M, reliable = np.eye(3), False
        self.homography, self.quality = None, 'none'
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
                    # Forward/backward check: a correspondence tracked back must land where it started.
                    back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, moved, None)
                    if back is not None:
                        keep &= back_status.ravel().astype(bool) & (np.hypot(*(back-points).reshape(-1, 2).T) <= ROUND_TRIP_PX)
                    if keep.sum() >= 12:
                        matrix, inliers = cv2.estimateAffinePartial2D(points[keep], moved[keep], method=cv2.RANSAC, ransacReprojThreshold=2)
                        if matrix is not None and inliers is not None:
                            zoom = float(np.sqrt(abs(np.linalg.det(matrix[:, :2]))))
                            reliable = bool(inliers.mean() > .6 and .85 < zoom < 1.18 and np.linalg.norm(matrix[:, 2]) < 180)
                            if reliable:
                                M = np.vstack([matrix, [0, 0, 1]]).astype(np.float64)
                                M[:2, 2] /= scale
                                self.quality = 'flow'
                                # A panning camera moves the ground plane by a homography, which the similarity
                                # only approximates; composed over seconds of panning the approximation drifts,
                                # so the pitch model is carried by the full homography when enough background
                                # points support one.
                                if keep.sum() >= 40:
                                    Hm, mask = cv2.findHomography(points[keep], moved[keep], cv2.RANSAC, 3.0)
                                    if Hm is not None and mask is not None and mask.mean() > .6 and _sane_homography(Hm, WORK_W, gray.shape[0]):
                                        S = np.diag([scale, scale, 1.0])
                                        self.homography = np.linalg.inv(S) @ (Hm/Hm[2, 2]) @ S
        if not reliable:
            shift = self.coarse_shift(small)
            if shift is not None:
                # The coarse pan: a translation in full-frame pixels, no zoom or rotation.
                M = np.eye(3)
                M[0, 2], M[1, 2] = shift[0]*w/COARSE_W, shift[1]*w/COARSE_W
                reliable, self.quality = True, 'coarse'
        cut = bool(self.thumbnail is not None and not reliable and float(np.mean(cv2.absdiff(thumbnail, self.thumbnail))) > 65)
        self.previous, self.thumbnail, self.coarse = gray, thumbnail, small
        return M, reliable, cut


class AnchorViews:
    """The frames calibration keyframes (and automatic anchors) were taken on, remembered as SIFT features
    with the homography of that frame, so that a later frame can be registered to them directly: a chain
    of frame-to-frame motions drifts by metres within seconds on low amateur cameras, a direct match to the
    keyframe's view does not while enough of it stays in view. People and the top band are masked as in
    the motion estimate; everything works at WORK_W width."""

    def __init__(self, width, height, limit=REGISTER_VIEWS, relays=RELAY_VIEWS):
        self.width, self.height = width, height
        self.scale = WORK_W/width
        self.views = {}
        self.limit = limit
        self.relays = relays
        self.sift = None
        self.matcher = None

    def _tools(self):
        if self.sift is None:
            self.sift = cv2.SIFT_create(nfeatures=REGISTER_FEATURES)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def features(self, gray, person_boxes):
        """Keypoints and descriptors of a WORK_W-wide grey frame, people and the top band masked."""
        self._tools()
        mask = np.full(gray.shape, 255, np.uint8)
        mask[:round(gray.shape[0]*.25)] = 0
        for x, y, bw, bh in person_boxes:
            cv2.rectangle(mask, (int(x*gray.shape[1]), int(y*gray.shape[0])), (int((x+bw)*gray.shape[1]), int((y+bh)*gray.shape[0])), 0, -1)
        keypoints, descriptors = self.sift.detectAndCompute(gray, mask)
        if descriptors is None or len(keypoints) < REGISTER_MIN_MATCHES:
            return None, None
        return np.float32([k.pt for k in keypoints]), descriptors

    def remember(self, index, time, gray, person_boxes, H, observability, root, relay=False):
        """Keep this frame as a view to register to. H: world metres -> full-resolution pixels of this frame.
        A relay is a re-registered frame kept while the keyframe's own view leaves the frame: later frames
        register to it, rooted at the same keyframe, with far fewer hops than the frame-to-frame chain."""
        points, descriptors = self.features(gray, person_boxes)
        if points is None:
            return False
        self.views[index] = {'index': index, 'time': time, 'points': points, 'descriptors': descriptors, 'H': np.asarray(H, np.float64).copy(),
                             'observability': float(observability), 'root': root, 'shape': gray.shape, 'relay': relay}
        for kind, limit in ((True, self.relays), (False, self.limit)):
            while sum(1 for v in self.views.values() if v['relay'] == kind) > limit:
                del self.views[min(i for i, v in self.views.items() if v['relay'] == kind)]
        return True

    def register(self, gray, person_boxes, prefer=None):
        """Register the current frame to the remembered views: {'index', 'H' (anchor-frame pixels -> current
        pixels, full resolution), 'matches', 'rms', 'overlap', 'relay'} of the view to use, or None. Keyframe
        views come first (the preferred one, then the most recent) and one that still matches well is kept; only
        when none does is the best view, relays included, used."""
        if not self.views:
            return None
        points, descriptors = self.features(gray, person_boxes)
        if points is None:
            return None
        order = sorted(self.views, key=lambda i: (self.views[i]['relay'], i != prefer, -i))
        best = None
        for index in order:
            view = self.views[index]
            candidate = self._match(view, points, descriptors, gray.shape)
            if candidate is None:
                continue
            if not view['relay'] and candidate['matches'] >= 2*REGISTER_MIN_MATCHES:
                return candidate
            if best is None or candidate['matches'] > best['matches']:
                best = candidate
        return best

    def _match(self, view, points, descriptors, shape):
        pairs = self.matcher.knnMatch(view['descriptors'], descriptors, k=2)
        src, dst = [], []
        for pair in pairs:
            if len(pair) == 2 and pair[0].distance < REGISTER_RATIO*pair[1].distance:
                src.append(view['points'][pair[0].queryIdx])
                dst.append(points[pair[0].trainIdx])
        if len(src) < REGISTER_MIN_MATCHES:
            return None
        src, dst = np.float32(src), np.float32(dst)
        Hm, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if Hm is None or mask is None or not np.isfinite(Hm).all() or abs(Hm[2, 2]) < 1e-9:
            return None
        Hm = Hm/Hm[2, 2]
        inliers = mask.ravel().astype(bool)
        n = int(inliers.sum())
        if n < REGISTER_MIN_MATCHES or np.linalg.det(Hm[:2, :2]) <= 0 or abs(Hm[2, 0]) > 1e-3 or abs(Hm[2, 1]) > 1e-3:
            return None
        rms = float(np.sqrt((np.hypot(*(transform_points(Hm, src[inliers])-dst[inliers]).T)**2).mean()))
        if rms > REGISTER_MAX_RMS:
            return None
        h, w = shape
        corners = transform_points(Hm, [(0, 0), (w, 0), (w, h), (0, h)])
        overlap = _overlap(corners/[w, h])
        if overlap < REGISTER_MIN_OVERLAP:
            return None
        S = np.diag([self.scale, self.scale, 1.0])
        return {'index': view['index'], 'H': np.linalg.inv(S) @ Hm @ S, 'matches': n, 'rms': rms, 'overlap': overlap, 'relay': view['relay']}


def _overlap(polygon):
    """Share of the unit frame covered by a (convex) normalized polygon."""
    from .geometry import clip_unit, polygon_area
    clipped = clip_unit([(float(x), float(y)) for x, y in polygon])
    return polygon_area(clipped) if len(clipped) >= 3 else 0.0


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
        self.anchored = None  # on that update: (points in pixels, rms) of the keyframe, for the offline registration
        self.anchored_time = None   # ... and the keyframe's own time

    @property
    def H(self):
        """Image pixels -> pitch metres while the pitch model is trusted (MEDIUM or better), else None."""
        if self.pitch.H is None or self.pitch.level == LOW_LEVEL:
            return None
        try:
            return np.linalg.inv(self.pitch.H)
        except np.linalg.LinAlgError:
            return None

    def update(self, timestamp, M, reliable, cut=False, homography=None, quality='flow'):
        """homography: the frame-to-frame homography when the camera motion estimate found one; it
        carries the pitch model more faithfully than the similarity M through long pans. quality: how
        the motion was measured ('flow', 'coarse', 'none'); a coarse pan decays the model faster."""
        dt = max(0.0, timestamp-self.last_time) if self.last_time is not None else 0.0
        self.last_time = timestamp
        self.fresh, self.anchored = False, None
        if self.polygon is not None:
            if reliable:
                self.polygon = transform_points(M, self.polygon)
            else:
                self.polygon = None
                self.warning = 'Field boundary lost on an unreliable camera move; add another boundary keyframe.'
        had = self.pitch.H is not None
        self.pitch.carry(homography if homography is not None and reliable else M, reliable, cut, dt, timestamp, quality)
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
                    # Up to KEYFRAME_MAX_RMS metres of landmark disagreement anchor the model: the residual lowers the
                    # observability (field.observability) instead of a hard limit dropping the keyframe silently.
                    solved = calibration.solve_details(anchor['points'], self.width, self.height, max_error=KEYFRAME_MAX_RMS, size=self.size)
                    H, self.rms, points = solved['H'], solved['rms'], solved['points']
                    observability = calibration.field.observability({'points': points}, self.width, self.height, size=self.size, rms_m=self.rms)
                    if solved['outliers']:
                        observability['outliers'] = solved['outliers']
                    if self.pitch.anchor(H, timestamp, self.rms, observability=observability['score'], details=observability):
                        self.warning, self.fresh, self.anchored, self.anchored_time = None, True, (points, self.rms), anchor['time']
                        if solved['outliers']:
                            self.warning = f'Calibration at {anchor["time"]:.2f}s: {", ".join(solved["outliers"])} disagreed with the other landmarks and was left out'
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
