// @ts-check
import { defineConfig } from 'astro/config';
import { loadEnv } from 'vite';
import {
  buildProxyConfig,
  describeProxy,
  readAllowedHosts,
  readProxyEnvironment,
} from './proxy/gateway-proxy.mjs';

// The proxy's two knobs come from `.env` (git-ignored) or the shell. `loadEnv`
// with an empty prefix reads BOTH — Astro/Vite only auto-populate the
// client-exposed namespaces (`PUBLIC_`/`VITE_`), and the key deliberately sits
// outside them, so it would otherwise never be read at all. Nothing from this
// record reaches a bundle: it is consumed here, in the Node process, to build
// server-side proxy options. See site/proxy/gateway-proxy.mjs for the full
// rationale and site/README.md for the operator flow.
const env = loadEnv('development', process.cwd(), '');
const proxyEnvironment = readProxyEnvironment(env);
const allowedHosts = readAllowedHosts(env);

// Printed once at config load so `npm run dev` states plainly where it will
// forward and whether a credential is attached — the answer to "why am I
// getting a 401" without anyone reaching for the key itself. Presence only,
// never the value.
console.log(describeProxy(proxyEnvironment));

// https://astro.build/config
export default defineConfig({
  // Pure static output — no adapter, no SSR. `astro build` emits a static
  // dist/ that makes zero external network requests (fonts are self-hosted
  // via @fontsource-variable, see src/layouts/Layout.astro).
  //
  // This site RUNS LOCALLY by decision (issue #151 scope boundary): `astro
  // dev` on the box next to the gateway, driving the /v1/realtime WebSocket
  // surface from a browser. There is no `site` URL to declare (the org config
  // sets one for its Cloudflare Pages deploy; this project has no deploy
  // target) and no adapter is ever added here. An operator MAY make that
  // local dev server reachable from anywhere by fronting it with a tunnel and
  // an SSO gate (Cloudflare Tunnel + Access) — still the same local process,
  // so nothing here changes except which Host headers it answers (below).
  output: 'static',

  vite: {
    server: {
      // The local credential-injecting proxy (issue #151 t13). This is what
      // lets a browser reach a header-authenticated gateway at all: the page
      // holds no key and dials only its own origin, and this Node process
      // attaches `Authorization: Bearer <key>` on the way out.
      //
      // DEV-SERVER ONLY, by construction. `astro build` emits static files and
      // `astro preview` serves them WITHOUT this proxy — a built site opened
      // from anywhere but `npm run dev` cannot reach the gateway. That is the
      // intended failure mode, not a gap: the site runs locally and has no
      // deploy path (no workflow under .github/workflows publishes it, and
      // none should be added).
      proxy: buildProxyConfig(proxyEnvironment),
      // Extra Host headers to answer, from LOBES_SITE_ALLOWED_HOSTS in the
      // git-ignored .env — the public hostname of an operator's SSO-gated
      // tunnel. Unset leaves Vite's loopback-only default untouched.
      ...(allowedHosts.length > 0 ? { allowedHosts } : {}),
    },
  },
});
