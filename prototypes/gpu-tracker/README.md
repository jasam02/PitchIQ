# PitchIQ local GPU prototype

A separate, native Windows review tool for a local soccer video. The soccer
detector runs on CUDA; tracking, optical flow, and appearance embeddings use
the CPU. Your footage stays on your computer. No account, upload, or API credits
are needed. The main PitchIQ application does not need to be running.

## Run from Command Prompt

The environment and detector are already installed on this computer. From the
repository root:

```bat
cd /d C:\Users\some1\Documents\PitchIQ\PitchIQ
prototypes\gpu-tracker\start.cmd "C:\Users\some1\Downloads\Video Project 1.mp4"
```

Open http://127.0.0.1:8765 and click **Process on GPU**. Previous results for the
same video load automatically. Add `--run` to the command to start processing
immediately, or `--port 8766` to use another port. Ctrl+C stops the server.

Choose a segment up to 180 seconds long. Longer files can be reviewed in
separate segments; identities are not linked between runs. The initial 1280
setting combines 960 and 1280 whole-frame passes: the lower-resolution pass
retains nearby players that this checkpoint sometimes misses at higher
resolutions. Choose 960 for a single, faster pass. Extra small-ball scans inspect
overlapping crops every other frame and take longer.

Play or step through the original video with overlays. Click a track to jump
to its first observation and highlight it. A label such as T9 is a temporary
track ID, **not jersey number 9**. Kit A/B are learned from torso colors across
tracks; display colors are arbitrary. Uncertain team membership stays unknown
and does not remove a detected player.

For sidelines or neighboring matches, pause at the start of a clear view and
draw a polygon around the visible playing area, then save that keyframe. Add
more after large camera changes and rerun. Background optical flow propagates
the boundary through pans and zooms. If motion estimation fails, filtering
is disabled until the next keyframe so it does not silently discard players.
These polygons are image-space exclusion regions, not field calibration.

## Install on another computer

Requires Windows, Python 3.12, an NVIDIA GPU, an appropriate NVIDIA driver,
and several GB of disk space. From the repository root:

```bat
prototypes\gpu-tracker\setup.cmd
```

Setup creates an isolated `.venv` and installs PyTorch 2.8/CUDA 12.8 for the
RTX 5080, then downloads the checksum-pinned soccer checkpoint. If Python 3.12
is not registered with the Python launcher, set `PITCHIQ_PYTHON` to its full
`python.exe` path. On this machine setup can also use Codex's bundled Python.
The existing `public/models/osnet-x025.onnx` must be present in the checkout.
No global Python packages are changed.

## Pipeline and limits

- A soccer-fine-tuned YOLOv8x checkpoint detects players, goalkeepers, referees,
  and ball candidates at each source frame.
- BoT-SORT associates people using motion, camera-motion compensation, and
  OSNet appearance features. OSNet is a general pedestrian model, so similarly
  dressed players can still exchange IDs. This does not solve long disappearances.
- Two torso-color groups are inferred after processing, with uncertain tracks
  left unassigned. Colors are not hard-coded and are not a strict detection gate.
- Ball selection uses confidence, camera-adjusted continuity, and ambiguity
  rejection. It emits no invented positions during gaps. Small balls and balls
  in neighboring games remain difficult; every ball output is provisional.
- Off-screen players remain unknown. There is no jersey OCR, roster identity,
  automatic metric pitch calibration, distance/speed reporting, full-match
  identity repair, or coaching model in this first prototype.

This is an initial runnable pipeline, not an implementation of the full
Broadcast2Pitch research system. Review its behavior on league footage before
using derived statistics. Ball coverage is not an accuracy measurement, and
track-segment count is not unique-player count.

Runs are saved under `runs/<video-content-hash>/<run>/`, including settings,
tracking JSON, and the latest preview. The original video is played directly;
an annotated MP4 is not generated. The viewer can download the tracking JSON.
The server binds only to loopback. Environments, downloaded weights, settings,
and generated runs are ignored by Git. Source changes remain uncommitted.

## Model provenance and licensing

- Detector: [gianpaj/football-players-detection-1](https://huggingface.co/gianpaj/football-players-detection-1).
  Exact revision, size, and SHA-256 are recorded in `model.json` and verified
  before loading. Its model card declares AGPL-3.0. Its training data is the
  [Roboflow football-players-detection dataset](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc).
- Detector framework and tracker: [Ultralytics v8.3.203](https://github.com/ultralytics/ultralytics/tree/v8.3.203),
  under its AGPL-3.0 / enterprise licensing options.
- Appearance model: existing [OSNet model notes](../../public/models/OSNET-MODEL.md)
  and [MIT notice](../../public/models/OSNET-LICENSE.txt).

Local use has no per-video API fee, but software/model license obligations still
apply. This prototype is not a blanket clearance to distribute these components
inside a closed-source commercial PitchIQ release.
