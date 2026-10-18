"""Re-decode the final ball trajectory of an existing result (soccer/ball_path.py) without re-running
the detector: the regression tool for ball tracking on real footage.

    python ball_cleanup.py runs/<video>/<run>/result.json [--video uploads/<video>.mp4] [--motion motion.npz]
                           [--registration registration.json] [--small-scene auto|on|off]
                           [--out result.cleaned.json] [--dry]

Camera motion per frame comes from the run's registration.json (the carried homographies the pipeline's own decode
used; the first pass's nearTrack and the registration timeline the pipeline mapped through are not stored, so a
re-decode matches the stored path closely, not bit for bit), else from the result itself (`frames[].motion`, the
per-frame estimate rounded to 5 decimals: still things drift in it), else from --motion (an .npz with M, reliable,
cut), else it is re-estimated from the video. --small-scene: the frames' own small-scene gate (auto,
`frames[].profile`), or forced on every frame for results written before the profile existed (on / off), for the
decode only: the written result keeps the profiles the causal pass recorded. Prints the
first-pass and final ball metrics, the still-lock audits and the first-pass sightings the final path leaves out.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from soccer import ball_path, field, registration


def motions_of(result, args, path):
    frames = result['frames']
    reg = Path(args.registration) if args.registration else path.with_name('registration.json')
    if reg.exists():
        record = json.loads(reg.read_text(encoding='utf-8'))
        offset = int(record.get('offset', 0))
        log = record['frames'][offset:offset+len(frames)]
        if len(log) == len(frames):
            return [(None if e['M'] is None else np.asarray(e['M'], np.float64).reshape(3, 3), bool(e['reliable']), bool(e['cut'])) for e in log]
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


def finalize(frames, motions, W, H, fps, small_scene='auto'):
    """ball_path.finalize, with the small-scene gate forced on every frame (small_scene 'on' / 'off') for this decode only:
    the frames keep the profile their causal pass recorded, which the other offline passes (tracklets, labels,
    imputation) trust."""
    if small_scene == 'auto':
        return ball_path.finalize(frames, motions, W, H, fps, to_pitch_of(frames))
    saved = [(f['profile'],) if 'profile' in f else () for f in frames]
    for f in frames:
        f['profile'] = {**(f.get('profile') or {}), 'smallScene': small_scene == 'on'}
    try:
        return ball_path.finalize(frames, motions, W, H, fps, to_pitch_of(frames))
    finally:
        for f, kept in zip(frames, saved):
            if kept:
                f['profile'] = kept[0]
            else:
                f.pop('profile', None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('result')
    parser.add_argument('--video')
    parser.add_argument('--motion')
    parser.add_argument('--registration', help="the run's registration.json (default: next to the result, when it exists)")
    parser.add_argument('--small-scene', choices=('auto', 'on', 'off'), default='auto')
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
    motions = motions_of(result, args, path)
    report = finalize(frames, motions, W, H, fps, args.small_scene)
    print('first pass:', json.dumps(report['firstPass']))
    print('final:     ', json.dumps(report['final']))
    for a in report['audits']:
        rivals = ', '.join(f"{r['tracklet']} (detector {r['det']:.2f}, {r['speed']:.1f} m/s)" for r in a['competitors'])
        print(f"  lock audit {a['start']:.2f}-{a['end']:.2f} s: tracklet {a['tracklet']} ({a['sightings']} causal sightings, moved {a['moved']:.2f} m, "
              f"entered {a['entered']}) gave way to {rivals}")
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
