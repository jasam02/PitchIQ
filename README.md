# PitchIQ — browser tracking workspace

An experimental [local NVIDIA GPU prototype](prototypes/gpu-tracker/README.md)
also provides soccer-trained detection, camera-aware tracking, inferred kit
groups, and a standalone video review page. It runs separately from this browser
workspace and is set up for the RTX 5080. See its README for Command Prompt steps.

The workflow uploads MP4s in small chunks and runs **YOLOX detection, OSNet appearance Re-ID and persistent soccer tracking in the browser**, with Norfair motion filtering in a worker. Clients do not need Python. Open a match, let its first frame scan, confirm clear player identities, mark referees with **Track as referee**, and set goalkeeper roles. Teams can also be inferred from repeated kit evidence; automatically allocated roster slots are provisional identities, not recognized jersey numbers.

The ball trail shows recent observations, compensated for camera movement. It stops at uncertain gaps and camera cuts; it is not a prediction of the next pass and does not itself improve the detector.

Keep the tab open while tracking. The upload limit is 150 MiB. Tracking saves every five processed seconds and archives older observations in immutable chunks, so the former short-clip point cap no longer stops the main tracking pipeline. Processing remains device-dependent and slower than playback; it is not an unattended background service. The COCO detector can miss tiny/occluded players and balls and cannot read names or jersey numbers. See [persistent soccer tracking](docs/SOCCER_TRACKING.md) for the identity architecture, limitations, and evaluation workflow.

See [September 29 update](docs/SEPT29_TRACKING_UPDATE.md) for changes, evidence and limits. The optional [native runner](services/video/README.md) remains for developers, but is no longer part of the client workflow.

## Run locally

Install Node.js 22.13 or later (Node 22 LTS is the baseline), Git, and the project's pinned pnpm:

```sh
npm install -g pnpm@11.25.0
pnpm install --frozen-lockfile
pnpm db:local
pnpm dev
```

Open http://localhost:5173 and upload an MP4 directly. No sign-in is required. This version uses one shared workspace; everyone who can access this app can access its footage. Keep this development server local. PitchIQ accounts will replace the temporary workspace identity in `lib/server.ts`; the existing local owner ID is retained to preserve local data. Do not use `pnpm install:ci` for normal laptop setup: that script expects the original managed environment.

The local database and video bucket are emulated by Cloudflare tooling and stored beneath `.wrangler/`. No Cloudflare credentials or AI API key are required for local development. Uploaded footage and saved matches from the hosted site are NOT in this export; upload a short MP4 locally to test. The demo analytics are seeded, not calculated from footage.

```sh
pnpm test
pnpm typecheck
pnpm test:norfair
pnpm build
```

Production hosting is a separate task: this is a full-stack Worker application, not a static GitHub Pages site. See the handoff before deploying elsewhere.

The first `pnpm dev` or `pnpm build` prepares ~37 MB of checksum-pinned browser runtime assets. No Python installation is required. Production serves these assets from the site itself; match footage is not sent to a third-party tracker.

## What's included

- React/TypeScript UI, backend routes, schema and database migration.
- Browser Norfair integration, player/ball trackers, appearance memory and timestamped field references.
- YOLOX-Tiny model and ONNX Web runtime binaries, with their licenses.
- Lockfile, tracking/upload regression tests, real-runtime tests, numeric detection fixture and benchmark script.
- Earlier roadmap and release history, including explicit prototype limitations.

The GitHub copy excludes the live Sites project identifier, installed dependencies, build outputs, runtime state, credentials and user-uploaded videos. Runtime assets are restored by the build script.

## License

No new license is granted for the original project code by this export; the owner should decide licensing before wider redistribution. Preserve the bundled third-party notices (`public/models`, `public/ort`, `public/norfair`, `build`, and `vendor`). npm dependencies retain their own licenses.
