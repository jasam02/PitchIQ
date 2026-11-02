"""The journal the fake workers (fullmatch_fake_worker.py) keep of every job start and end, one JSON line per event,
shared by all of them and read by the scheduler tests.

Several worker processes append to it at once, and each reads it to count a job's attempts. Appends from several
processes are not atomic on Windows and a reader can catch a line another process is still writing: under load a fake
worker read half a line, crashed, and a scheduler test's match failed for it (2 of 15 runs with the CPUs busy, once in
the full suite). So every append and every read holds an OS lock on journal.lock beside it; the system releases the
lock when a process dies holding it (some fake workers die on purpose).
"""
import contextlib
import json
import os
import time
from pathlib import Path

if os.name == 'nt':
    import msvcrt
else:
    import fcntl

WAIT_S = 20.0


@contextlib.contextmanager
def locked(journal):
    with open(Path(journal).with_name('journal.lock'), 'a+b') as handle:
        deadline = time.time()+WAIT_S
        while True:
            try:
                if os.name == 'nt':
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.time() > deadline:
                    raise
                time.sleep(.002)
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def append(journal, event):
    with locked(journal), Path(journal).open('a', encoding='utf-8') as out:
        out.write(json.dumps(event)+'\n')


def read(journal):
    journal = Path(journal)
    if not journal.exists():
        return []
    with locked(journal):
        text = journal.read_text(encoding='utf-8')
    return [json.loads(line) for line in text.splitlines()]
