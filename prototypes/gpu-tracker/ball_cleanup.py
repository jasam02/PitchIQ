"""Re-decode the final ball trajectory of an existing result (soccer/ball_path.py) without re-running
the detector: the regression tool for ball tracking on real footage.

    python ball_cleanup.py runs/<video>/<run>/result.json [--video uploads/<video>.mp4] [--motion motion.npz]
                           [--out result.cleaned.json] [--dry]

Camera motion per frame comes from the result itself (`frames[].motion`, written by current runs), else
from --motion (an .npz with M, reliable, cut), else it is re-estimated from the video. Prints the
first-pass and final ball metrics and the first-pass sightings the final path leaves out.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from soccer import ball_path, field, registration


def motions_of(result, args):
    frames = result['frames']
    if all('motion' in f for f in frames):
        out = []
        for f in frames:
            m = f.get('motion')
            M = np.array([[m[0], m[1], m[2]], [m[3], m[4], m[5]], [0, 0, 1]], float) if m else None
            out.append((M, bool(f.get('cameraReliable')) and M is not None, bool(f.get('cut'))))
        return out
    if args.motion:
        data = np.load(args.motion)
        return [(M, bool(r), bool(c)) for M, r, c in zip(data['M'], data['reliable'], data['cut'])]
    if not args.video:
        raise SystemExit('This result stores no camera motion: pass --motion <npz> or --video <file> to re-estimate it.')
    import cv2
    from soccer.camera import CameraMotion
    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0]['frame'])
    cam, previous, out = CameraMotion(), [], []
    for f in frames:
        ok, img = cap.read()
        if not ok:
            break
        M, reliable, cut = cam.estimate(img, previous)
        out.append((M if reliable else None, reliable, cut))
        previous = [p['box'] for p in f['people']]
    return out+[(None, False, False)]*(len(frames)-len(out))


def to_pitch_of(frames):
    """The offline registration stored per frame (`calibration.H`, world metres -> pixels, and its confidence) as the
    `to_pitch(i, point_px)` the decoder uses to recognize the painted spots; None when no frame carries one."""
    size = field.PitchSize()
    for f in frames:
        sz = (f.get('pitchModel') or {}).get('size')
        if sz and sz.get('length'):
            size.length, size.width = float(sz['length']), float(sz['width'])
            break
    Hs = []
    for f in frames:
        cal = f.get('calibration') or {}
        h = cal.get('H')
        Hs.append((np.array(h, float).reshape(3, 3), float(cal.get('confidence') or 0.0)) if h and len(h) == 9 else None)
    if not any(Hs):
        return None

    def to_pitch(i, point):
        if i >= len(Hs) or Hs[i] is None:
            return None
        p, _, _ = registration.position(Hs[i][0], size, point, Hs[i][1])
        return p
    return to_pitch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('result')
    parser.add_argument('--video')
    parser.add_argument('--motion')
    parser.add_argument('--out')
    parser.add_argument('--dry', action='store_true', help='print the report only')
    args = parser.parse_args()
    path = Path(args.result)
    result = json.loads(path.read_text(encoding='utf-8'))
    frames = result['frames']
    for f in frames:   # start from the causal pass when the result was cleaned before
        if isinstance(f.get('ball'), dict) and f['ball'].get('firstPass') and f['ball']['firstPass'].get('center'):
            fp = f['ball']['firstPass']
            f['ball'] = {'state': fp['state'], 'phase': fp.get('phase'), 'confidence': fp.get('confidence', 0.0), 'track': fp.get('track'), 'center': fp['center'],
                         'box': f['ball'].get('box', [0, 0, 0, 0])}
        elif isinstance(f.get('ball'), dict) and f['ball'].get('firstPass'):
            f['ball'] = {'state': 'UNKNOWN', 'confidence': 0.0}
    W, H, fps = result['video']['width'], result['video']['height'], result['video']['fps']
    motions = motions_of(result, args)
    report = ball_path.finalize(frames, motions, W, H, fps, to_pitch_of(frames))
    print('first pass:', json.dumps(report['firstPass']))
    print('final:     ', json.dumps(report['final']))
    for d in report['dropped']:
        print(f"  dropped {d['start']:.2f}-{d['end']:.2f} s ({d['frames']} frames" + (f", on track {d['player']}'s lower body" if d['player'] is not None else '') + f"): {d['why']}")
    if args.dry:
        return
    summary = result.setdefault('summary', {}).setdefault('ball', {})
    summary.update({'trackedShare': report['final']['states']['TRACKED'], 'missingShare': report['final']['states']['MISSING'],
                    'unknownShare': report['final']['states']['UNKNOWN'], 'teleports': report['final']['teleports'],
                    'footSwitches': report['final']['footSwitches'], 'tracks': report['final']['runs'],
                    'longestUnknownSeconds': report['final']['longestUnknownSeconds'], 'droppedFirstPassRuns': len(report['dropped']),
                    'firstPass': {**summary.get('firstPass', {}), **report['firstPass']}})
    result.setdefault('events', []).extend(report['events'])
    out = Path(args.out) if args.out else path.with_name(path.stem+'.cleaned.json')
    out.write_text(json.dumps(result, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    print('written', out)


if __name__ == '__main__':
    main()
