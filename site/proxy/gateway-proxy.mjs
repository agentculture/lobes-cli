/**
 * The local credential-injecting proxy — issue #151 t13.
 *
 * WHY THIS EXISTS
 * ---------------
 * The lobes gateway gates every `/v1/*` route — the `/v1/realtime` WebSocket
 * handshake included — on an `Authorization: Bearer <key>` HEADER, checked
 * with `hmac.compare_digest` before the realtime branch is even reached
 * (`lobes/gateway/server.py` `_Handler.do_GET`). A browser `WebSocket` cannot
 * set arbitrary headers, and query-parameter / `Sec-WebSocket-Protocol`
 * authentication are OUT OF SCOPE by operator decision (spec boundary: "the
 * public gateway remains header-authenticated" — public realtime clients are
 * robots and native applications that set the header natively).
 *
 * So the browser never talks to the gateway. It talks to its OWN origin (the
 * Astro dev server), and this proxy — running in that same Node process —
 * attaches the credential on the way out. The key lives in exactly one
 * process's environment and is never serialised into anything the browser
 * receives: not HTML, not JS, not a config endpoint, not a query string.
 *
 * That mirrors the discipline the gateway itself applies one hop later: it
 * DROPS the caller's `Authorization` before relaying the handshake to the
 * realtime bridge (`_DROP_FROM_HANDSHAKE` in `lobes/gateway/_realtime.py`),
 * because a credential is spent the moment it is validated. We hold the same
 * bar in the other direction — the browser is never handed one to spend.
 *
 * WHICH MECHANISM, AND WHY
 * ------------------------
 * This is the Astro/Vite dev-server proxy (`server.proxy`, `ws: true`), the
 * option the spec prefers because it needs no second process. The plan
 * tracked "Vite ws upgrade-header injection is unverified on this Vite major"
 * as an open risk with a standalone `ws` proxy as the pinned fallback. It was
 * verified on the installed Vite 8.1.5 (bundled `http-proxy-3@1.23.3`) and it
 * WORKS, so the fallback was not built — see site/README.md, "Why the Vite
 * dev-server proxy". The relevant guarantees, read off the bundled
 * `ws-incoming` pass:
 *
 *   - `proxyReqWs` fires after `setupOutgoing` and BEFORE `proxyReq.end()`,
 *     so `setHeader`/`removeHeader` land on the wire.
 *   - `setupSocket` puts `setTimeout(0)` on both legs, so an idle listening
 *     session is never torn down by a proxy timeout — a realtime session is
 *     silent by design between utterances.
 *
 * A NOTE ON CORS: the gateway sends no `Access-Control-Allow-*` headers at
 * all, so browser HTTP to it is cross-origin-blocked regardless of auth.
 * Everything the page fetches therefore goes through this same-origin proxy.
 */

/** Where the proxy forwards when `LOBES_GATEWAY_URL` is unset. */
export const DEFAULT_GATEWAY_URL = "http://127.0.0.1:8000";

/** The env var naming the gateway origin to forward to. */
export const GATEWAY_URL_VAR = "LOBES_GATEWAY_URL";

/**
 * The env var holding the bearer credential.
 *
 * Deliberately carries NEITHER the `PUBLIC_` prefix (Astro's client-exposed
 * namespace) NOR the `VITE_` prefix (Vite's). Both toolchains inline
 * prefix-matching variables into client bundles; a key named with either
 * prefix would be shipped to the browser by the build itself. This name is
 * inert to both.
 */
export const GATEWAY_API_KEY_VAR = "LOBES_GATEWAY_API_KEY";

/**
 * Browser-visible paths this proxy claims on the dev-server origin.
 *
 * `/v1/` covers both the WebSocket session route (`/v1/realtime`) and the
 * keyed HTTP routes the connection panel preflights against (`/v1/models`),
 * so the page addresses the fleet at EXACTLY the paths the real API uses —
 * only the origin differs. `/capabilities` is keyless on the gateway but
 * still needs proxying, because CORS (see above) blocks it cross-origin.
 */
export const PROXY_CONTEXTS = /** @type {const} */ (["/v1/", "/capabilities"]);

/**
 * @typedef {object} ProxyEnvironment
 * @property {string} gatewayUrl Origin the proxy forwards to.
 * @property {string} apiKey Bearer credential, or `""` when the operator set none.
 */

/**
 * Read the proxy's two knobs out of an environment-shaped record.
 *
 * Pure — takes the record rather than touching `process.env` — so the
 * resolution rules are unit-testable without mutating global state.
 *
 * An absent key is NOT an error. The gateway's own inbound bearer gate is
 * opt-in (`GATEWAY_API_KEY` unset → the gate returns before the header is
 * even read), so a bare local fleet is reached with no credential at all.
 * Demanding one here would break the commonest local setup.
 *
 * @param {Record<string, string | undefined>} env
 * @returns {ProxyEnvironment}
 */
export function readProxyEnvironment(env) {
  const rawUrl = (env[GATEWAY_URL_VAR] ?? "").trim();
  const rawKey = (env[GATEWAY_API_KEY_VAR] ?? "").trim();
  return {
    gatewayUrl: rawUrl === "" ? DEFAULT_GATEWAY_URL : stripTrailingSlash(rawUrl),
    apiKey: rawKey,
  };
}

/**
 * @param {string} value
 * @returns {string}
 */
function stripTrailingSlash(value) {
  return value.endsWith("/") ? value.slice(0, -1) : value;
}

/**
 * Headers this proxy strips from every forwarded request.
 *
 * The browser's cookies for `localhost:4321` mean nothing to the gateway and
 * have no business in its logs — the same reasoning `_DROP_FROM_HANDSHAKE`
 * applies to the caller's credential one hop later. `authorization` is in the
 * list too: whatever the page might somehow have sent is discarded, and the
 * value THIS process sets is the only one that ever reaches the gateway.
 */
const STRIPPED_REQUEST_HEADERS = ["cookie", "authorization"];

/**
 * Build the `vite.server.proxy` record.
 *
 * @param {ProxyEnvironment} environment
 * @returns {Record<string, import("vite").ProxyOptions>}
 */
export function buildProxyConfig({ gatewayUrl, apiKey }) {
  /** @type {import("vite").ProxyOptions} */
  const shared = {
    target: gatewayUrl,
    changeOrigin: true,
    // Both the session WebSocket and the preflight GETs ride the same entry:
    // one target, one credential rule, no chance of the two drifting apart.
    ws: true,
    configure: (proxy) => {
      // `proxyReq` is the HTTP pass, `proxyReqWs` the upgrade pass. Both fire
      // before the request is flushed, so both can rewrite headers. Wiring
      // only one of them is the silent half-failure this hook exists to
      // avoid: the preflight would authenticate and the session would 401,
      // or the reverse.
      proxy.on("proxyReq", (proxyReq) => applyCredential(proxyReq, apiKey));
      proxy.on("proxyReqWs", (proxyReq) => applyCredential(proxyReq, apiKey));
    },
  };

  /** @type {Record<string, import("vite").ProxyOptions>} */
  const config = {};
  for (const context of PROXY_CONTEXTS) {
    config[context] = shared;
  }
  return config;
}

/**
 * Strip inbound credentials, then attach the operator's — if there is one.
 *
 * Split out and exported so the ordering (strip, THEN set) is asserted by a
 * test rather than by reading the callback: a hook that set the header first
 * and stripped second would ship an unauthenticated handshake.
 *
 * @param {{ removeHeader: (name: string) => void, setHeader: (name: string, value: string) => void }} proxyReq
 * @param {string} apiKey
 */
export function applyCredential(proxyReq, apiKey) {
  for (const header of STRIPPED_REQUEST_HEADERS) {
    proxyReq.removeHeader(header);
  }
  if (apiKey !== "") {
    proxyReq.setHeader("Authorization", `Bearer ${apiKey}`);
  }
}

/**
 * One line for the dev-server banner: where requests go and whether a
 * credential is attached.
 *
 * Reports only the PRESENCE of a key — never a prefix, a length, or a hash.
 * A dev-server log is a file, a scrollback buffer, and sometimes a pasted
 * bug report; "a key is set" is the whole of what an operator needs to
 * diagnose a 401, and it is the whole of what this prints.
 *
 * @param {ProxyEnvironment} environment
 * @returns {string}
 */
export function describeProxy({ gatewayUrl, apiKey }) {
  const credential =
    apiKey === ""
      ? `no credential (${GATEWAY_API_KEY_VAR} unset — correct for a gateway with GATEWAY_API_KEY unset)`
      : `injecting Authorization: Bearer <${GATEWAY_API_KEY_VAR}>`;
  return `[lobes] /v1/* + /capabilities -> ${gatewayUrl}; ${credential}`;
}
