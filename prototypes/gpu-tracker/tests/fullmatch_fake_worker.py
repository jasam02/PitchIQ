"""A stand-in for fullmatch/worker.py in the scheduler tests: the same JSON-lines protocol, no models, no GPU.

    python fullmatch_fake_worker.py SCRIPT.json --name worker-N [--light]

SCRIPT.json: {"delay": seconds per job, "startup": "ok" | "fail", "outcomes": {"<segment index>": ["oom", "ok"], ...},
"chunkOutcomes": {"<discovery chunk>": ["error", "ok"], ...}, "chunkDelay": seconds}.
Each attempt of a segment takes the next outcome from its list (then "ok"): ok (a verified artifact like the real
worker's: result.json covering the processed frames, kept compressed as result.json.gz, identities.json,
registration.json, segment.json last, renamed into place), error, oom, crash (the process dies mid-job), slow (ok after a long delay, cancellable). Every job start and end
is appended to the script's journal (journal.jsonl next to it, under a lock the processes share: fullmatch_journal.py),
so a test can count runs and check concurrency. A
discovery job ({"kind": "discover"}) writes rows for its sampled frames (no anchors) to the job's file, like the real
worker; a light worker (--light) takes those only.
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from fullmatch import manifest  # noqa: E402
import fullmatch_journal  # noqa: E402  (this script's own folder is first on the path)

script_path = Path(sys.argv[1])
name = sys.argv[sys.argv.index('--name')+1] if '--name' in sys.argv else 'worker'
light = '--light' in sys.argv
script = json.loads(script_path.read_text(encoding='utf-8'))
journal = script_path.with_name('journal.jsonl')
lock = threading.Lock()
cancel = threading.Event()
jobs = []
gone = threading.Event()
out = sys.stdout


def send(message):
    with lock:
        out.write(json.dumps(message)+'\n')
        out.flush()


def note(**fields):
    with lock:
        fullmatch_journal.append(journal, fields | {'worker': name, 'pid': os.getpid(), 't': time.time()})


def read():
    for line in sys.stdin:
        message = json.loads(line)
        if message['type'] == 'cancel':
            cancel.set()
        else:
            jobs.append(message)
    gone.set()
    cancel.set()


def outcome(index, event='start', key='segment', table='outcomes'):
    """The outcome of this attempt: attempts are counted in the journal, so a replacement process continues the list."""
    with lock:
        events = fullmatch_journal.read(journal)
    n = sum(1 for e in events if e['event'] == event and e.get(key) == index)
    sequence = script.get(table, {}).get(str(index), [])
    return sequence[n] if n < len(sequence) else 'ok'


def discover(job):
    what = outcome(job['chunk'], 'chunk-start', 'chunk', 'chunkOutcomes')
    note(event='chunk-start', chunk=job['chunk'], outcome=what, light=light)
    deadline = time.time()+float(script.get('chunkDelay', .05))*(30 if what == 'slow' else 1)
    while time.time() < deadline and not cancel.is_set():
        time.sleep(.01)
    if cancel.is_set():
        note(event='chunk-cancelled', chunk=job['chunk'])
        send({'type': 'failed', 'job': job['key'], 'kind': 'cancelled', 'jobKind': 'discover', 'chunk': job['chunk'], 'error': 'cancelled'})
        return
    if what == 'crash':
        note(event='chunk-crash', chunk=job['chunk'])
        os._exit(3)
    if what == 'error':
        send({'type': 'failed', 'job': job['key'], 'kind': 'error', 'jobKind': 'discover', 'chunk': job['chunk'], 'error': 'ValueError: fake'})
        return
    rows = [{'index': i, 'time': i/25.0, 'ok': False, 'reason': 'fake'} for i in job['indices']]
    manifest.write_json(Path(job['out']), {'rows': rows, 'pairs': [], 'chunk': job['chunk']})
    note(event='chunk-end', chunk=job['chunk'])
    send({'type': 'done', 'job': job['key'], 'kind': 'discover', 'chunk': job['chunk'], 'frames': len(rows), 'anchors': 0, 'elapsed': .05})


if script.get('startup') == 'fail':
    send({'type': 'failed', 'job': None, 'kind': 'startup', 'error': 'RuntimeError: no GPU (fake)'})
    sys.exit(2)
threading.Thread(target=read, daemon=True).start()
send({'type': 'ready', 'pid': os.getpid(), 'name': name, 'loadSeconds': 0.01, 'light': light})
while not gone.is_set():
    if not jobs:
        time.sleep(.01)
        continue
    message = jobs.pop(0)
    if message['type'] == 'shutdown':
        break
    job = message['job']
    cancel.clear()
    if job.get('kind') == 'discover':
        discover(job)
        continue
    if light:
        send({'type': 'failed', 'job': job['key'], 'kind': 'error', 'error': 'a light worker runs no segment jobs'})
        continue
    what = outcome(job['index'])
    note(event='start', segment=job['index'], outcome=what, calibrations=[c['time'] for c in job['config'].get('calibrations') or []])
    delay = float(script.get('delay', .1))*(30 if what == 'slow' else 1)
    deadline = time.time()+delay
    while time.time() < deadline and not cancel.is_set():
        send({'type': 'progress', 'job': job['key'], 'stage': 'Tracking', 'progress': .5, 'frames': (job['proc1']-job['proc0'])//2})
        time.sleep(min(.05, delay))
    if not cancel.is_set():
        # The offline passes: stage messages without a frame count (the real worker's pipeline sends these).
        send({'type': 'progress', 'job': job['key'], 'stage': 'Decoding the final ball trajectory', 'frames': None})
        time.sleep(min(.05, delay))
    if cancel.is_set():
        note(event='cancelled', segment=job['index'])
        send({'type': 'failed', 'job': job['key'], 'kind': 'cancelled', 'error': 'cancelled'})
        continue
    if what == 'crash':
        note(event='crash', segment=job['index'])
        os._exit(3)
    if what in ('error', 'oom'):
        note(event=what, segment=job['index'])
        send({'type': 'failed', 'job': job['key'], 'kind': what, 'error': 'CUDA out of memory (fake)' if what == 'oom' else 'ValueError: fake'})
        continue
    partial = Path(job['partial'])
    partial.mkdir(parents=True)
    result = {'status': 'complete', 'frames': [{'frame': f, 'time': f/job['fps'], 'people': []} for f in range(job['proc0'], job['proc1'])]}
    (partial/'result.json').write_text(json.dumps(result), encoding='utf-8')
    (partial/'identities.json').write_text('{}', encoding='utf-8')
    (partial/'registration.json').write_text('{}', encoding='utf-8')
    stored, _ = manifest.compress_result(partial)
    facts = manifest.sidecar(result, stored)
    manifest.write_json(partial/manifest.SIDECAR, facts)
    os.replace(partial, job['folder'])
    note(event='end', segment=job['index'])
    send({'type': 'done', 'job': job['key'], 'sidecar': facts, 'elapsed': delay, 'summary': {}})
send({'type': 'bye', 'pid': os.getpid()})
