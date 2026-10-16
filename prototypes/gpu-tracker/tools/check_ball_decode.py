"""Side-by-side of the vectorized offline ball decoder against the frozen one-frame-at-a-time copy
(tests/ball_path_reference.py) on the frames of a real run:

    python tools/check_ball_decode.py runs/<key>/<run>            # a run folder with result.json (and registration.json)
    python tools/check_ball_decode.py C:/scratch/A/result.json

The layers are rebuilt from the result's ballCandidates exactly as pipeline.finalize_ball builds them (nodes,
attachment, own speeds, standing markings); the camera motions come from registration.json when it sits next to
the result (the carried homographies the run used), else from the frames' own motion rows. Prints the time each
decode took and whether the paths are the same node for node, with the same link costs and speeds bit for bit.
Exit code 1 when they differ."""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests'))

from soccer import ball_path, pitch_model, registration  # noqa: E402
import ball_synth  # noqa: E402
from ball_path_reference import decode_reference  # noqa: E402


def motions_of(folder, frames):
    record = folder/'registration.json'
    if record.exists():
        data = json.loads(record.read_text(encoding='utf-8'))
        offset = int(data.get('offset', 0))
        log = data['frames'][offset:offset+len(frames)]
        if len(log) == len(frames):
            return [(None if e['M'] is None else np.asarray(e['M'], np.float64).reshape(3, 3), bool(e['reliable']), bool(e['cut'])) for e in log], 'registration.json'
    out = []
    for f in frames:
        m = f.get('motion')
        M = None if m is None else np.array([[m[0], m[1], m[2]], [m[3], m[4], m[5]], [0.0, 0.0, 1.0]])
        out.append((M, bool(f.get('cameraReliable', M is not None)), bool(f.get('cut', False))))
    return out, 'the frames\' motion rows'


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    path = Path(argv[1])
    folder = path if path.is_dir() else path.parent
    result = json.loads((folder/'result.json').read_text(encoding='utf-8'))
    frames = result['frames']
    W, H, fps = result['video']['width'], result['video']['height'], float(result['video']['fps'])
    size = pitch_model.field.PitchSize.from_json((result.get('tactical') or {}).get('pitchSize')) if (result.get('tactical') or {}).get('pitchSize') else pitch_model.field.PitchSize()
    motions, source = motions_of(folder, frames)

    def to_pitch(i, point):
        c = frames[i].get('calibration') or {}
        if not c.get('H'):
            return None
        p, _, _ = registration.position(np.asarray(c['H'], np.float64).reshape(3, 3), size, point, c.get('confidence', 0.0))
        return p
    layers = [ball_path.nodes_of(i, f, W, H, to_pitch) for i, f in enumerate(frames)]
    ball_path._attachment(layers, fps)
    times = [f['time'] for f in frames]
    T, _, _ = ball_path._chain(motions)
    ball_path._velocities(layers, times, T)
    ball_path._standing_markings(layers, times, T)
    sizes = [len(l) for l in layers]
    print(f'{len(frames)} frames, {sum(sizes)} candidates ({np.mean(sizes):.1f} a frame, at most {max(sizes)}), motions from {source}')
    t0 = time.perf_counter()
    expected = decode_reference(layers, times, motions, fps)
    t_ref = time.perf_counter()-t0
    t0 = time.perf_counter()
    actual = ball_path.decode(layers, times, motions, fps)
    t_new = time.perf_counter()-t0
    same, why = ball_synth.same_path(expected, actual)
    print(f'reference decode {t_ref:.2f} s, vectorized decode {t_new:.2f} s ({t_ref/max(t_new, 1e-9):.1f}x); '
          f'{sum(1 for p in actual if p is not None)} frames on the path')
    print('identical:', 'yes' if same else f'NO ({why})')
    return 0 if same else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
