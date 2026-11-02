"""Full Match Mode's frame timing: the worker turns a segment's frames [proc0, proc1) into the start and length the
pipeline takes (fullmatch/worker.window_times) from the decoder's own timestamps, which are irregular on real files.

A stand-in decoder reports timestamps that are not frame/fps: a constant offset (CAN-ISL: -11.9 ms) that jumps part way
through (later -18.6 ms), and a variant whose timestamps run ahead of frame/fps. What is checked: the pipeline's own
frame selection (pipeline.run: it seeks to round(start*fps), treats frames earlier than start as pre-roll and stops at
the first frame at or after start+seconds, or at the duration frames/fps) applied to that start and length processes
exactly frames proc0..proc1-1, before and after the jump and at the end of the video; and the manifest's verification
lets the last segment of a period end up to TAIL_TOLERANCE frames short (timestamps ahead of frame/fps stop the run at
the duration), nowhere else.

Run:  python -m unittest test_fullmatch_timing          (from this directory)
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from fullmatch import manifest as mf, plan, worker  # noqa: E402

FPS = 25.00008
N = 20000


def stamps_lagging():
    return [i/FPS-(.0119 if i < 12000 else .0186) for i in range(N)]


def stamps_leading():
    return [i/FPS+.015 for i in range(N)]


class FakeCapture:
    """cv2.VideoCapture over a table of timestamps: set(POS_FRAMES), read() advances, get(POS_MSEC) is the time of
    the frame just read (as OpenCV's FFmpeg backend reports it)."""
    table = []

    def __init__(self, path):
        self.i = 0

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.i = int(value)
        return True

    def read(self):
        if self.i >= len(self.table):
            return False, None
        self.i += 1
        return True, object()

    def get(self, prop):
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.table[self.i-1]*1000.0
        return 0.0

    def release(self):
        pass


def pipeline_frames(stamps, start, seconds, frames, fps):
    """The frames pipeline.run emits for a start and length (its Prefetch rules, without the post-roll)."""
    finish = min(frames/fps, start+seconds)
    first = int(round(start*fps))
    out = []
    for i in range(first, frames):
        t = stamps[i]
        if t < start-1e-6:
            continue          # pre-roll
        if t >= finish:
            break
        out.append(i)
    return out


class WindowTimes(unittest.TestCase):
    def check(self, stamps, proc0, proc1):
        FakeCapture.table = stamps
        with mock.patch.object(cv2, 'VideoCapture', FakeCapture):
            start, seconds = worker.window_times('x.mp4', proc0, proc1, N, FPS)
        return pipeline_frames(stamps, start, seconds, N, FPS)

    def test_irregular_timestamps_give_exactly_the_planned_frames(self):
        stamps = stamps_lagging()
        for s in plan.plan(N, FPS, plan.periods(N/FPS)):
            got = self.check(stamps, s.proc0, s.proc1)
            self.assertEqual((got[0], got[-1], len(got)), (s.proc0, s.proc1-1, s.proc1-s.proc0), s)

    def test_timestamps_ahead_of_the_frame_rate_lose_at_most_the_tail(self):
        stamps = stamps_leading()
        segments = plan.plan(N, FPS, plan.periods(N/FPS))
        for s in segments[:-1]:
            got = self.check(stamps, s.proc0, s.proc1)
            self.assertEqual((got[0], got[-1]), (s.proc0, s.proc1-1))
        last = segments[-1]
        got = self.check(stamps, last.proc0, last.proc1)
        self.assertEqual(got[0], last.proc0)
        self.assertGreaterEqual(got[-1], last.own1-1-mf.TAIL_TOLERANCE)


class TailTolerance(unittest.TestCase):
    def facts(self, first, last):
        return {'status': 'complete', 'contiguous': True, 'first': first, 'last': last}

    def test_only_a_periods_last_segment_may_end_short(self):
        import json
        import tempfile
        from pathlib import Path
        segments = plan.plan(N, FPS, plan.periods(N/FPS))
        middle, last = segments[1], segments[-1]
        # A middle segment may lose its context tail (it still covers what it owns), never an owned frame.
        for s, short, ok in ((last, 20, True), (last, 60, False), (middle, middle.proc1-middle.own1, True),
                             (middle, middle.proc1-middle.own1+5, False)):
            folder = Path(tempfile.mkdtemp())
            result = {'status': 'complete', 'frames': [{'frame': f, 'time': f/FPS} for f in range(s.proc0, s.proc1-short)]}
            (folder/'result.json').write_text(json.dumps(result), encoding='utf-8')
            for name in ('identities.json', 'registration.json'):
                (folder/name).write_text('{}', encoding='utf-8')
            mf.write_json(folder/mf.SIDECAR, mf.sidecar(result, folder/'result.json'))
            problem = mf.verify_artifact(folder, s)
            self.assertEqual(problem is None, ok, (s.index, short, problem))


if __name__ == '__main__':
    unittest.main()
