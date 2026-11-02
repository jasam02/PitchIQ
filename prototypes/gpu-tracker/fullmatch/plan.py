"""The segment plan of a match: which frames of the source each segment job processes and which frames it owns.

A match is the whole source or the stretch the user marks, in one or more periods: the first and the second half
when a halftime break is marked (extra time adds periods), else one. Each period is cut into segments of at most
SEGMENT_S seconds of video, equal in length, that share the period's frames out without a gap or an overlap: the
owned frames [own0, own1), frame indices of the source, half-open. Every segment also processes PAD_S seconds more
on each side, inside its period (the processed frames [proc0, proc1)): the context the start of a run lacks (new
identities, the ball lock and the small-scene gate need a second or two) and that its offline passes use for the
owned frames near its ends. The frames two neighbours both process (2 x PAD_S) are where the stitching compares them:
the same people, the ball, the calibration. Only the owned frames of a segment go into the match, so nothing is
counted twice. Nothing crosses a period boundary: no context, no stitching, no trajectory across halftime.

Frame indices, not timestamps, are the identity of a frame: decoders report irregular timestamps (CAN-ISL: frame
10896 at 435.828111 s, -11.9 ms off frame/fps, later -18.6 ms) and seeking to a frame index is exact (checked on that
file, pixels and timestamps). Times in a plan are frame/fps, for display and estimates; the worker turns a segment's
frames into the start and length the pipeline takes by reading the timestamps of the frames themselves.
"""
import math
from bisect import bisect_right
from dataclasses import asdict, dataclass

SEGMENT_S = 180.0    # seconds of owned video per segment at most (the short-segment limit; a run's memory grows with it)
PAD_S = 3.0          # seconds of context on each side of the owned frames (inside the period)
MIN_PERIOD_S = 5.0   # a period shorter than this is refused (a slip of the halftime marker)
MAX_PERIODS = 4      # two halves and two halves of extra time


@dataclass(frozen=True)
class Segment:
    index: int    # 0.. over the whole match
    period: int   # 0.. index of its period
    own0: int     # first owned frame
    own1: int     # one past the last owned frame
    proc0: int    # first processed frame (own0 minus the context, inside the period)
    proc1: int    # one past the last processed frame

    @property
    def key(self):
        """A stable name for the segment's artifacts: index and owned frames, so a different plan never reuses them."""
        return f's{self.index:03d}-f{self.own0}-{self.own1}'

    def to_json(self, fps):
        out = asdict(self)
        out['key'] = self.key
        out['ownStart'], out['ownEnd'] = round(self.own0/fps, 3), round(self.own1/fps, 3)
        out['procStart'], out['procEnd'] = round(self.proc0/fps, 3), round(self.proc1/fps, 3)
        return out

    @classmethod
    def from_json(cls, data):
        return cls(*(int(data[k]) for k in ('index', 'period', 'own0', 'own1', 'proc0', 'proc1')))


def to_frame(t, fps, frames):
    """The frame index nearest to time t (seconds, frame/fps), inside [0, frames]."""
    return min(max(int(round(float(t)*fps)), 0), int(frames))


def periods(duration, start=None, end=None, halftime=None, ranges=None):
    """The periods of a match, in seconds [(t0, t1), ...]: `ranges` when given (each half's own range, extra time),
    else the stretch [start, end] (default the whole source) split by an optional halftime break (b0, b1). Raises
    ValueError for ranges outside the video, overlapping or out of order, or shorter than MIN_PERIOD_S."""
    duration = float(duration)
    if ranges:
        out = [(float(a), float(b)) for a, b in ranges]
    else:
        t0 = 0.0 if start is None else float(start)
        t1 = duration if end is None else float(end)
        out = [(t0, t1)]
        if halftime is not None:
            b0, b1 = (float(v) for v in halftime)
            if not t0 < b0 <= b1 < t1:
                raise ValueError('The halftime break must lie inside the match range.')
            out = [(t0, b0), (b1, t1)]
    if not 1 <= len(out) <= MAX_PERIODS:
        raise ValueError(f'A match has 1 to {MAX_PERIODS} periods.')
    previous = 0.0
    for a, b in out:
        if not all(math.isfinite(v) for v in (a, b)) or a < 0 or b > duration+1e-6:
            raise ValueError('Every period must lie inside the video.')
        if a < previous-1e-6:
            raise ValueError('Periods must not overlap and must be in order.')
        if b-a < MIN_PERIOD_S:
            raise ValueError(f'A period must be at least {MIN_PERIOD_S:.0f} seconds long.')
        previous = b
    return [(a, min(b, duration)) for a, b in out]


def plan(frames, fps, period_ranges, segment_s=SEGMENT_S, pad_s=PAD_S):
    """The segments of a match: `frames` and `fps` of the source, `period_ranges` in seconds (periods()). Each period
    of F frames becomes n = ceil(F / segment frames) segments of F/n frames each (rounded to whole frames), so the last
    one is never a sliver; each processes up to pad_s more on each side inside its period."""
    fps, frames = float(fps), int(frames)
    if not fps > 0 or frames <= 0:
        raise ValueError('The video has no usable frame timing.')
    if not segment_s >= 10 or pad_s < 0:
        raise ValueError('Segments must be at least 10 s and the context not negative.')
    step = max(1, int(round(segment_s*fps)))
    pad = int(round(pad_s*fps))
    out = []
    for p, (t0, t1) in enumerate(period_ranges):
        f0, f1 = to_frame(t0, fps, frames), to_frame(t1, fps, frames)
        if f1 <= f0:
            raise ValueError('A period has no frames.')
        n = -(-(f1-f0)//step)
        bounds = [f0+round(i*(f1-f0)/n) for i in range(n+1)]
        for a, b in zip(bounds, bounds[1:]):
            out.append(Segment(len(out), p, a, b, max(f0, a-pad), min(f1, b+pad)))
    check(out)
    return out


def check(segments):
    """The plan's invariants, raised as ValueError: indices in order; within a period the owned frames follow each other
    without a gap or an overlap; the processed frames contain the owned ones and stay inside the period's owned span;
    periods do not overlap."""
    by_period = {}
    for i, s in enumerate(segments):
        if s.index != i:
            raise ValueError('Segment indices must run 0, 1, 2, ...')
        if not (s.proc0 <= s.own0 < s.own1 <= s.proc1):
            raise ValueError(f'Segment {i}: the processed frames must contain the owned ones.')
        by_period.setdefault(s.period, []).append(s)
    spans = []
    for p in sorted(by_period):
        group = by_period[p]
        for a, b in zip(group, group[1:]):
            if b.own0 != a.own1:
                raise ValueError(f'Segments {a.index} and {b.index} leave a gap or overlap.')
        lo, hi = group[0].own0, group[-1].own1
        if any(s.proc0 < lo or s.proc1 > hi for s in group):
            raise ValueError(f'Period {p}: a segment processes frames outside its period.')
        spans.append((lo, hi))
    if any(b[0] < a[1] for a, b in zip(spans, spans[1:])):
        raise ValueError('Periods overlap.')
    return True


def owner(segments, frame):
    """The segment that owns this frame, or None (outside every period: before, after, or in the halftime break)."""
    starts = [s.own0 for s in segments]
    i = bisect_right(starts, int(frame))-1
    if i >= 0 and segments[i].own0 <= frame < segments[i].own1:
        return segments[i]
    return None


def neighbours(segments):
    """Pairs (a, b) of segments next to each other in one period: the boundaries the stitching works on."""
    return [(a, b) for a, b in zip(segments, segments[1:]) if a.period == b.period and a.own1 == b.own0]


def processed_frames(segments):
    """Frames processed over the whole plan, context included (the work an estimate counts)."""
    return sum(s.proc1-s.proc0 for s in segments)
