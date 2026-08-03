# Muscle Memory frontend

This directory is the single browser product surface for Muscle Memory. The root route is the
product overview, and its primary action opens the live operator workspace at `/console`.
`/about` remains a compatibility alias for the overview. The production Vite build
is served by the FastAPI process, so the cloud deployment has one origin for HTTP, WebSocket, and UI
traffic.

## Locked stack

- React 19 and TypeScript 6
- Vite 8 for development and production builds
- Motion for React for entrance, scroll, and pointer-responsive animation
- Three.js for the scroll-directed household scene and MM-01 demonstration
- Lucide React for interface icons
- Self-hosted Manrope and Newsreader fonts
- Plain CSS with shared design tokens; no runtime CSS framework
- npm with exact package versions recorded in `package-lock.json`

Node 22 or newer is required. The dependency versions in `package.json` are exact rather than
ranges; update them deliberately and commit the regenerated lockfile together.

## Commands

```bash
npm install
npm run dev
npm run lint
npm run build
npm run preview
```

The development server binds to `0.0.0.0:4173` and proxies `/api` (including WebSocket upgrades) to
`http://127.0.0.1:8000`. A production build is discovered at `frontend/dist` by FastAPI; set
`MM_FRONTEND_DIST` only when the build is stored elsewhere.

## Product rules

- The landing hero renders locally with WebGL; its two-story house still is captured from the same
  checked-in scene and never depends on a network image.
- Pointer motion is transform-only and has a `prefers-reduced-motion` fallback.
- Every breakpoint must remain horizontally scroll-free and leave the next section visible below
  the hero.
- Provider, episode, and policy states must come from backend contracts. A disconnected backend is
  rendered as unavailable, never replaced with plausible production data.
- Sensor views preserve all eight categories and their required use labels when the operator
  workspace is connected.
- Generated visual assets are appearance-only. Physics state and collider truth come from backend
  records, not the browser scene.

## Runtime contracts

The workspace reads `/api/v1/health`, `/episodes`, `/approvals/pending`, `/policies`, and the
selected episode's telemetry or replay page. Running episodes attach to
`/api/v1/episodes/{episode_id}/live`; the browser accepts only the typed 20 Hz messages and keeps
`frame_id` as the sole video join. The live controls read `/api/v1/live/options`, select only an
admitted catalog seed and evaluated policy, and use authenticated start/cancel mutations. Camera
surfaces use the direct 30 FPS MJPEG URLs during a run and exact-frame URLs while scrubbing a
buffered replay. Otherwise they remain visibly unavailable; the browser does not substitute the
illustrative product scene for simulator output.

Human decisions and route or keep-out corrections use the API's existing authenticated mutation
routes. The operator credential is held in `sessionStorage`, never written into a URL, build asset,
or persistent browser storage. Held-out policy statistics and promotion state are rendered only
from `/policies` and `/policies/promotion-eligibility`; action agreement or development metrics are
never relabeled as held-out success.
