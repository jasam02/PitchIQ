"""The ball regression clips of the 0-60 s broadcast window, read from a result: what the final ball path
and the identities did around the times the earlier passes got wrong.

    python tools/ball_clips.py C:/scratch/speed-pass-3/timing/after/result.json
    python tools/ball_clips.py result.json --window 12.0 15.0 --window 40.8 42.8

Default windows (the broadcast upload 2bac432528d0e568, start 0):
  12.0-15.0 s  the path must not sit on the right penalty spot at 12.57-13.37 s (flagged: pitch position within
               1.5 m of (94/105, .5), or no pitch position while the image point is far from every tracked player)
  22.1-24.3 s  the path must not visit the ball boy's ball on the bottom touchline (flagged: pitch y beyond 1.0 or
               the image centre in the bottom 8% of the frame)
  40.8-42.8 s  B-07 goes behind A-03 and must come out as B-07 (the frames list both labels, and any identity that
               first appears in the window)
Each window prints one line per frame: time, state / phase, origin and motion, image centre in pixels, pitch
position, the player within reach, and the flags; then a one-line verdict per window."""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from soccer.geometry import PAINTED_SPOTS  # noqa: E402

DEFAULT = [(12.0, 15.0, 'penalty spot'), (22.1, 24.3, 'ball boy'), (40.8, 42.8, 'B-07 behind A-03')]
RIGHT_SPOT = PAINTED_SPOTS[1]


def flags(frame, ball, W, H, size):
    out = []
    p = ball.get('pitch')
    if ball.get('state') == 'UNKNOWN':
        return out
    if p is not None:
        metres = math.hypot((p[0]-RIGHT_SPOT[0])*size[0], (p[1]-RIGHT_SPOT[1])*size[1])
        if metres <= 1.5:
            out.append(f'ON THE RIGHT PENALTY SPOT ({metres:.1f} m)')
        if p[1] > 1.0 or p[1] < 0.0:
            out.append('BEYOND A TOUCHLINE')
    c = ball.get('center')
    if c is not None and c[1] >= .92:
        out.append('BOTTOM OF THE FRAME')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('result')
    ap.add_argument('--window', nargs=2, type=float, action='append', metavar=('START', 'END'))
    a = ap.parse_args(argv)
    result = json.loads(Path(a.result).read_text(encoding='utf-8'))
    frames = result['frames']
    W, H = result['video']['width'], result['video']['height']
    psize = (result.get('tactical') or {}).get('pitchSize') or {'length': 105.0, 'width': 68.0}
    size = (float(psize['length']), float(psize['width']))
    windows = [(s, e, '') for s, e in a.window] if a.window else DEFAULT
    first_seen = {}
    for f in frames:
        for p in f['people']:
            if p.get('id') and p['id'] not in first_seen:
                first_seen[p['id']] = f['time']
    for start, end, note in windows:
        rows = [f for f in frames if start-1e-6 <= f['time'] <= end+1e-6]
        print(f'\n== {start:.2f}-{end:.2f} s {note} ({len(rows)} frames) ==')
        flagged = 0
        labels = {}
        for f in rows:
            b = f['ball']
            c = b.get('center')
            centre = f'x {c[0]*W:6.0f} y {c[1]*H:6.0f}' if c else '      -        -'
            p = b.get('pitch')
            pitch = f'pitch ({p[0]:.3f}, {p[1]:.3f})' if p else 'pitch -'
            fl = flags(f, b, W, H, size)
            flagged += bool(fl)
            near = b.get('nearTrack')
            shown = {q['track']: q.get('display') for q in f['people']}
            print(f"{f['time']:7.2f}  {b['state']:<8}{(b.get('phase') or ''):<10}{(b.get('origin') or ''):<14}{(b.get('motion') or ''):<11}{centre}  {pitch}"
                  + (f"  near T{near} ({shown.get(near, '?')})" if near is not None else '') + ('  <-- '+'; '.join(fl) if fl else ''))
            for q in f['people']:
                d = q.get('display')
                if d in ('B-07', 'A-03') or (q.get('id') and start-1e-6 <= first_seen.get(q['id'], -1) <= end+1e-6):
                    labels.setdefault(d, []).append((f['time'], q['track'], q.get('id')))
        if labels:
            for d, seen in sorted(labels.items(), key=lambda kv: str(kv[0])):
                tracks = sorted({t for _, t, _ in seen})
                ids = sorted({i for _, _, i in seen if i})
                new = [i for i in ids if start-1e-6 <= first_seen.get(i, -1) <= end+1e-6]
                print(f"  label {d}: {len(seen)} frames, tracks {tracks}, identities {ids}" + (f"  (NEW in this window: {new})" if new else ''))
        states = {}
        for f in rows:
            states[f['ball']['state']] = states.get(f['ball']['state'], 0)+1
        print(f"  verdict: {flagged} flagged frames; ball states {states}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
