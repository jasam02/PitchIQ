"""Collect hard negatives for a future ball verifier from a result: every confidently detected ball
candidate that the tracker rejected, cropped from the original frames with context, sorted by reason.

    python ball_negatives.py runs/<video>/<run>/result.json uploads/<video>.mp4 <out dir> [--min-det 0.5] [--context 3]

Writes <out dir>/<reason>/<time>_<index>.jpg (reason folders such as PLAYER_ATTACHED, NO_ENTRY, JUMP,
FIELD_LINE, STATIC, OTHER_OBJECT, TRAJECTORY, LOW_SCORE) plus positives/ for the sightings on the final
trajectory, and an index.json with the candidate's scores, the player it sat on and the frame. Each crop
is a square of (context x the candidate's size) around it, so a boot, a sock, a line or a spot is seen
with its surroundings, which is what a verifier needs. Nothing is trained here.
"""
import argparse
import json
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('result')
    parser.add_argument('video')
    parser.add_argument('out')
    parser.add_argument('--min-det', type=float, default=.5, help='detector confidence below which rejected candidates are not kept')
    parser.add_argument('--context', type=float, default=3.0, help='crop size as a multiple of the candidate box (at least 48 px)')
    parser.add_argument('--every', type=int, default=3, help='keep every n-th frame (consecutive crops are near-duplicates)')
    args = parser.parse_args()
    result = json.loads(Path(args.result).read_text(encoding='utf-8'))
    frames = result['frames']
    W, H = result['video']['width'], result['video']['height']
    out = Path(args.out)
    cap = cv2.VideoCapture(args.video)
    index = []
    counts = {}
    for k, f in enumerate(frames):
        if k % args.every:
            continue
        wanted = [(c, c.get('reason') or 'candidate') for c in f.get('ballCandidates', []) if c.get('status') != 'ball' and c.get('det', 0) >= args.min_det]
        b = f.get('ball') or {}
        if b.get('state') == 'TRACKED' and b.get('observed') and b.get('box'):
            wanted.append(({'box': b['box'], 'det': b.get('detector', b.get('visual', 1.0)), 'scores': {}, 'index': 0}, 'positives'))
        if not wanted:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, f['frame'])
        ok, img = cap.read()
        if not ok:
            break
        for c, reason in wanted:
            x, y, w, h = c['box']
            cx, cy, size = (x+w/2)*W, (y+h/2)*H, max(48.0, args.context*max(w*W, h*H))
            x0, y0 = int(max(0, cx-size/2)), int(max(0, cy-size/2))
            crop = img[y0:int(min(H, cy+size/2)), x0:int(min(W, cx+size/2))]
            if crop.size == 0:
                continue
            folder = out/reason
            folder.mkdir(parents=True, exist_ok=True)
            name = f"{f['time']:08.3f}_{c.get('index', 0)}.jpg"
            cv2.imwrite(str(folder/name), crop)
            counts[reason] = counts.get(reason, 0)+1
            index.append({'file': f'{reason}/{name}', 'time': f['time'], 'frame': f['frame'], 'det': c.get('det'), 'reason': reason,
                          'scores': c.get('scores', {}), 'player': c.get('player'), 'box': c['box']})
    out.mkdir(parents=True, exist_ok=True)
    (out/'index.json').write_text(json.dumps(index, indent=1), encoding='utf-8')
    print({k: v for k, v in sorted(counts.items())})


if __name__ == '__main__':
    main()
