"""Fetch the pinned public soccer checkpoint; no account or API key required."""
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent


def get_model():
    spec = json.loads((ROOT / 'model.json').read_text())
    target = ROOT / 'models' / 'football-yolov8x.pt'
    target.parent.mkdir(exist_ok=True)
    def valid(path):
        if not path.exists() or path.stat().st_size != spec['bytes']:
            return False
        with path.open('rb') as handle:
            return hashlib.file_digest(handle, 'sha256').hexdigest() == spec['sha256']
    if valid(target):
        return target
    url = f"https://huggingface.co/{spec['repository']}/resolve/{spec['revision']}/{spec['file']}"
    partial = target.with_suffix('.part')
    print('Downloading soccer detector (137 MB)...', flush=True)
    with urlopen(Request(url, headers={'User-Agent': 'PitchIQ-local-prototype'}), timeout=120) as response:
        with partial.open('wb') as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    if not valid(partial):
        raise RuntimeError('Model checksum mismatch. The incomplete download was not loaded.')
    partial.replace(target)
    return target


if __name__ == '__main__':
    print(get_model())
