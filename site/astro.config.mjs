// @ts-check
import { defineConfig } from 'astro/config';

// https://astro.build/config
export default defineConfig({
  // Pure static output — no adapter, no SSR. `astro build` emits a static
  // dist/ that makes zero external network requests (fonts are self-hosted
  // via @fontsource-variable, see src/layouts/Layout.astro).
  //
  // This site is LOCAL-ONLY by decision (issue #151 scope boundary): it
  // exists so a developer can drive the /v1/realtime WebSocket surface from
  // a browser against a local gateway. There is no `site` URL to declare
  // (the org config sets one for its Cloudflare Pages deploy; this project
  // has no deploy target) and no adapter is ever added here.
  output: 'static',

  // t13 (local proxy + dev flow) is expected to add a `server.proxy` block
  // here for the Astro/Vite dev-server WebSocket proxy (server.proxy with
  // ws: true + an upgrade-header hook that injects the Authorization
  // bearer server-side, per docs/specs/2026-07-21-realtime-voice-to-voice-astro-test-site-151.md).
  // Left unset here deliberately — t10 scaffolds the project, t13 owns the
  // proxy wiring and the fallback standalone-proxy decision.
  vite: {
    server: {
      // t13 mount point: dev-only WS proxy config lands here.
    },
  },
});
