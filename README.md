# PitchIQ — browser tracking workspace

The hosted workflow now uploads MP4s in small chunks and runs **Norfair 2.3.0 automatically inside a browser worker**, through Pyodide. Clients do not need Python, a terminal, or export/import steps. Open a match, let its first frame scan, assign one clear player from each team, optionally suggest matching teammates, and label the ball. Check provisional roster identities before tracking.

The ball trail shows recent observations, compensated for camera movement. It stops at uncertain gaps and camera cuts; it is not a prediction of the next pass and does not itself improve the detector.

This remains a segment-tracking prototype. Keep the tab open. The upload limit is 100 MiB, tracking saves every five processed seconds and pauses at its existing saved-point limit (roughly 30 seconds with 22 active tracks). It is not yet an unattended full-match service. The COCO detector still misses some tiny/occluded balls and cannot identify names or jersey numbers.

**Soccer identity tracking** is now the default automatic mode. It ignores spectators and bench staff outside the pitch, separates referees and goalkeepers from players, and keeps one stable identity per roster player across exits, returns and camera cuts. Uncertain identities are left open instead of guessed. A debug overlay and an identity-event log show why each decision was made. See [Soccer identity tracking](docs/SOCCER_TRACKING.md) for the architecture, statuses, debug workflow, configuration and limits. An **Upload video** button is always available in the dashboard top bar and sidebar.

See [September 29 update](docs/SEPT29_TRACKING_UPDATE.md) for changes, evidence and limits. The optional [native runner](services/video/README.md) remains for developers, but is no longer part of the client workflow.

## Run locally

Install Node.js 22.13 or later (Node 22 LTS is the baseline), Git, and the project's pinned pnpm:

```sh
npm install -g pnpm@11.25.0
pnpm install --frozen-lockfile
pnpm db:local
pnpm dev
```

Open http://localhost:5173. Click Sign in with ChatGPT if prompted. The existing portable-development plugin provides a **local test user**, not your real hosted account. It only works on localhost/loopback; keep this development server local. Do not use `pnpm install:ci` for normal laptop setup: that script expects the original managed environment.

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
