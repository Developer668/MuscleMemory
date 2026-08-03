# Muscle Memory frontend

This directory is the single browser product surface for Muscle Memory. New public pages and the
operator workspace belong here; Python services expose APIs and streams but do not render a second
competing UI.

## Locked stack

- React 19 and TypeScript 6
- Vite 8 for development and production builds
- Motion for React for entrance, scroll, and pointer-responsive animation
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

The development server defaults to `http://127.0.0.1:4173/` and moves to another port only when
that port is already occupied.

## Product rules

- The landing hero uses a checked-in project asset, never a network-dependent image.
- Pointer motion is transform-only and has a `prefers-reduced-motion` fallback.
- Every breakpoint must remain horizontally scroll-free and leave the next section visible below
  the hero.
- Provider, episode, and policy states must come from backend contracts. A disconnected backend is
  rendered as unavailable, never replaced with plausible production data.
- Sensor views preserve all eight categories and their required use labels when the operator
  workspace is connected.
- Generated visual assets are appearance-only. Physics state and collider truth come from backend
  records, not the browser scene.
