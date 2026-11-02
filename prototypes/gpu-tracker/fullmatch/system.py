"""What this computer can run: free memory, the GPU (nvidia-smi: present wherever the NVIDIA driver is, Windows or a Linux
container), CPU threads, disk, and from them the AUTO worker count.

The constants are measurements on the RTX 5080 machine (16 logical CPUs, 31 GB RAM), Oct 26-28 2026, CAN-ISL tactical
wide footage: a worker holds 2.5 GB RAM once its models are loaded and peaks at 3.0 GB over a 180 s segment (5.3 GB
committed); three concurrent workers used 4.7 GB of GPU memory with the desktop's 1.2 GB (about 1.2 GB each); aggregate
throughput was 7.5 / 11.7 / 14.6 frames per second for 1 / 2 / 3 concurrent segments (identical results). AUTO takes
the most workers memory, GPU memory and CPUs allow, at most AUTO_MAX. A fourth worker was 6.8% faster than three on the
same 12 x 40 s CAN-ISL segments (14.26 against 13.35 fps; 10.4 against 8.0 GB in the workers, 23.5 against 21.4 GB in
use, the GPU at 96% either way) but has not run a whole match, so it stays the user's experimental choice, never AUTO's.
"""
import math
import shutil
import subprocess

import psutil

WORKER_RAM_GB = 3.5      # per worker, with headroom over the measured 3.0 GB peak (three workers: 8.56 GB together, 12 min)
WORKER_VRAM_GB = 1.5     # per worker, with headroom over the measured ~1.2 GB
RESERVE_RAM_GB = 2.0     # left for the system and the browser (the server frees its own models, ~2 GB, when a match starts)
CPUS_PER_WORKER = 4      # logical CPUs a worker keeps busy (tracking thread, decode/detect thread, embedder threads)
AUTO_MAX = 3             # the default ceiling: 3 workers 14.6 fps aggregate on the 5080 (a fourth: above)
MAX_WORKERS = 4          # the experimental ceiling the page offers
UNSAFE_RAM_GB = 1.5      # available memory below this: start no new segment until it recovers
UNSAFE_VRAM_GB = .8      # free GPU memory below this: likewise
THROUGHPUT_FPS = {1: 7.5, 2: 11.7, 3: 14.6}   # measured aggregate fps (40 s window), the first estimate before a run measures its own
LONG_SEGMENT_FACTOR = .9                       # a 180 s segment ran ~10% slower than the 40 s window
BYTES_PER_FRAME = 25_000                       # result.json per processed frame (measured 17.6-25 KB on broadcast and CAN-ISL),
                                               # as a running segment writes it, before the worker compresses it
STORED_BYTES_PER_FRAME = 2_200                 # the result as a match keeps it (gzip): 1.2-2.0 KB measured
BYTES_PER_SEGMENT = 4_000_000                  # identities, registration, events, preview, logs per segment
ASSEMBLY_BYTES_PER_FRAME = 2_500               # the assembled match (views, samples, possession, coaching caches): 2.0-2.3 KB
                                               # measured (CAN-ISL whole match, dev12, the broadcast 0-303 s)
DISCOVERY_SECONDS_PER_FRAME = .37              # calibration discovery per sampled frame, one process, OpenCV single-threaded
                                               # (solve + SIFT + 3 pairs; CAN-ISL, a loaded machine, Oct 28: .27 s with its threads)
DISCOVERY_STEP_S = 2.0                         # its sampling, for the plan's estimate (calibration.STEP_S decides the run's)


def discovery_seconds(processed_frames, fps, streams):
    """Expected wall-clock seconds of the calibration discovery: the sampled frames on that many processes."""
    sampled = processed_frames/max(fps, 1e-6)/DISCOVERY_STEP_S
    return sampled*DISCOVERY_SECONDS_PER_FRAME/max(1, streams)


def gpu_status():
    """{'name', 'utilization', 'usedGB', 'totalGB', 'freeGB'} of the first GPU from nvidia-smi, or None."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=10)
        name, util, used, total = [v.strip() for v in out.stdout.strip().splitlines()[0].split(',')]
        used_gb, total_gb = float(used)/1024, float(total)/1024
        return {'name': name, 'utilization': float(util), 'usedGB': round(used_gb, 2), 'totalGB': round(total_gb, 2),
                'freeGB': round(total_gb-used_gb, 2)}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def gpu_processes():
    """{pid: GB} of the processes using the GPU (empty where nvidia-smi cannot tell, e.g. Windows WDDM shows N/A)."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=10)
        found = {}
        for line in out.stdout.strip().splitlines():
            pid, used = [v.strip() for v in line.split(',')]
            if used.replace('.', '', 1).isdigit():
                found[int(pid)] = round(float(used)/1024, 2)
        return found
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}


def snapshot(gpu=True):
    memory = psutil.virtual_memory()
    return {'ramTotalGB': round(memory.total/2**30, 1), 'ramAvailableGB': round(memory.available/2**30, 1),
            'cpus': psutil.cpu_count(logical=True) or 1, 'cpuPercent': psutil.cpu_percent(interval=None),
            'gpu': gpu_status() if gpu else None}


def auto_workers(snap):
    """(count, reasons): the AUTO worker count for this computer now, at least 1, at most AUTO_MAX."""
    limits = {'3-worker default': AUTO_MAX,
              'CPU threads': max(1, snap['cpus']//CPUS_PER_WORKER),
              'free memory': max(1, int((snap['ramAvailableGB']-RESERVE_RAM_GB)//WORKER_RAM_GB))}
    if snap.get('gpu'):
        limits['free GPU memory'] = max(1, int(snap['gpu']['freeGB']//WORKER_VRAM_GB))
    count = max(1, min(limits.values()))
    binding = [k for k, v in limits.items() if v == count]
    return count, {'limits': limits, 'binding': binding}


def allowed(requested, snap):
    """(count, note): the user's worker choice within safe limits (memory, GPU memory); note says what was reduced."""
    requested = max(1, min(int(requested), MAX_WORKERS))
    safe = requested
    by_ram = max(1, int((snap['ramAvailableGB']-RESERVE_RAM_GB)//WORKER_RAM_GB))
    safe = min(safe, by_ram)
    if snap.get('gpu'):
        safe = min(safe, max(1, int(snap['gpu']['freeGB']//WORKER_VRAM_GB)))
    note = None if safe == requested else f'{requested} workers do not fit in the free memory now; using {safe}.'
    return safe, note


def unsafe(snap):
    """Why starting another segment now would risk the machine (None when it is safe)."""
    if snap['ramAvailableGB'] < UNSAFE_RAM_GB:
        return f"only {snap['ramAvailableGB']:.1f} GB of memory is free"
    gpu = snap.get('gpu')
    if gpu and gpu['freeGB'] < UNSAFE_VRAM_GB:
        return f"only {gpu['freeGB']:.1f} GB of GPU memory is free"
    return None


def throughput(workers, measured=None):
    """Aggregate frames per second expected with this many workers: this machine's own measurement of an earlier match
    when there is one (measured: {count: fps}), else the 5080 benchmark scaled for 180 s segments."""
    if measured and str(workers) in measured:
        return float(measured[str(workers)])
    table = THROUGHPUT_FPS
    fps = table.get(workers) or table[max(table)]*(1+.1*(workers-max(table)))
    return fps*LONG_SEGMENT_FACTOR


def estimate_seconds(frames, workers, measured=None, overhead_s=0.0):
    """(low, likely, high) wall-clock seconds for this many processed frames: the likely value from the expected
    throughput, a +-20% range before the run has measured anything (more uncertainty: a different camera, a busy GPU)."""
    likely = frames/max(throughput(workers, measured), .1)+overhead_s
    return .8*likely, likely, 1.25*likely


def disk_needed(processed_frames, segments):
    """Bytes a match needs on disk (an estimate from the measured sizes): its results as it keeps them (compressed), the
    other artifact files, the assembled match, and room for the plain results of the segments that run at once (a
    worker compresses its result when its run ends)."""
    running = min(segments, MAX_WORKERS)*processed_frames/max(segments, 1)*BYTES_PER_FRAME
    return int(processed_frames*(STORED_BYTES_PER_FRAME+ASSEMBLY_BYTES_PER_FRAME)+segments*BYTES_PER_SEGMENT+running)


def disk_free(path):
    return shutil.disk_usage(path).free


def human_bytes(n):
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024 or unit == 'TB':
            return f'{n:.0f} {unit}' if unit in ('B', 'KB') else f'{n:.1f} {unit}'
        n /= 1024
    return f'{n:.1f} TB'


def ceil_div(a, b):
    return int(math.ceil(a/b))
