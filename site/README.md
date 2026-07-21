# lobes realtime test harness

A local-only Astro site for driving the fleet's `GET /v1/realtime` WebSocket
session from a browser: mic in, live event stream, audio out. It exists so the
realtime surface can be *experienced* — VAD boundaries, transcripts,
interruptions scrolling past in real time — instead of inferred from terminal
prints.

**It is never deployed.** There is no adapter, no `site:` URL, and no workflow
under `.github/workflows` that publishes it — and none should be added. The
only CI job that touches this directory builds it, so a broken site fails a PR;
nothing ships it anywhere. Issue #151 records that as a scope boundary, not an
oversight.

## Read this first: the microphone will silently not exist

The browser runs on **your laptop**. The fleet runs on a **headless box** (a
DGX Spark, a Jetson AGX Thor). Those are different machines, and that single
fact decides the whole dev flow:

> `getUserMedia` — the only way to reach a microphone — is gated on a **secure
> context**. A secure context means **HTTPS** or **`localhost`**. Nothing else.

So a site served at `http://spark:4321` has **no microphone at all**. Not a
denied permission, not an error dialog you can act on: `navigator.mediaDevices`
is simply `undefined`, and a page that does not check for that looks completely
fine and hears nothing forever. Every symptom points at VAD, at the model, at
the network — at anything but the URL bar.

There are exactly two ways out, and the first is the one to use.

### The flow: `ssh -L` (primary)

Forward both ports to your laptop, so the *browser's* idea of the world is
entirely `localhost`:

```bash
# On the laptop. 8000 = the gateway; the site runs locally against it.
ssh -N -L 8000:localhost:8000 you@spark
```

Then, in this directory on the laptop:

```bash
npm ci
cp .env.example .env      # then edit — see "Supplying the credential"
npm run dev               # http://localhost:4321
```

The page is `localhost` (secure context — microphone available) and the gateway
is `localhost:8000` (reached through the forward). Nothing is exposed to the
network, no certificate is involved, and the gateway is untouched.

If the site itself must run *on* the box (a slow link, a big model, whatever),
forward the site instead and open it as localhost anyway:

```bash
# On the laptop: 4321 = the Astro dev server running on the box.
ssh -N -L 4321:localhost:4321 you@spark
# on the box:
npm run dev -- --host 127.0.0.1
```

Either way the rule is the same: **the browser must see `localhost`**.

### The alternative: `mkcert` HTTPS

If forwarding is unacceptable, serve the site over real HTTPS with a locally
trusted certificate:

```bash
mkcert -install
mkcert lobes.local            # writes lobes.local.pem + lobes.local-key.pem
npm run dev -- --https --key ./lobes.local-key.pem --cert ./lobes.local.pem
```

Two things then have to hold, and both bite quietly:

- The **gateway** must also be reachable over HTTPS, or be forwarded to
  localhost. A `wss:` page cannot open a `ws:` socket — browsers block it as
  mixed content, and the report is unhelpful. (The site handles the scheme
  itself: a page on `https:` dials `wss:`, always.)
- Certificate files must never be committed. `.gitignore` covers `.env`; keep
  `*.pem` out of the tree yourself.

`ssh -L` avoids both. Prefer it.

## How the browser reaches a header-authenticated gateway

The gateway gates every `/v1/*` route — the `/v1/realtime` handshake
included — on an `Authorization: Bearer <key>` **header**, compared with
`hmac.compare_digest` before the realtime branch is even reached. A browser
`WebSocket` **cannot set request headers**. Query-parameter and
`Sec-WebSocket-Protocol` authentication are **out of scope by operator
decision**: public realtime clients are robots and native applications that set
the header natively, and the public gateway stays header-authenticated.

So the browser never talks to the gateway:

```text
browser ──same origin──▶ Astro dev server ──+ Authorization: Bearer ──▶ gateway
 (no key, ever)            (holds the key)                          (validates it)
```

The page dials **its own origin** (`/v1/realtime`, a path). The dev server
proxies that to the gateway and attaches the credential on the way out. The key
lives in exactly one process's environment. It is never in the HTML, never in
the JavaScript, never in a config endpoint, and never in a query string that
would land in every access log on the path.

This mirrors what the gateway does one hop later: it **drops** the caller's
`Authorization` before relaying the handshake to the realtime bridge, because a
credential is spent the moment it is validated. Same discipline, one hop
earlier — the browser is never handed one to spend. The proxy likewise strips
inbound `cookie` and `authorization` before forwarding, so browser cookies for
`localhost:4321` never reach the gateway's logs.

**The gateway is unchanged by any of this.** Nothing in `lobes/gateway/` moves
to make the browser work.

### Supplying the credential

Copy `.env.example` to `.env` (git-ignored) and set:

| Variable | Meaning | Default |
| --- | --- | --- |
| `LOBES_GATEWAY_URL` | Origin the proxy forwards to | `http://127.0.0.1:8000` |
| `LOBES_GATEWAY_API_KEY` | The gateway's `GATEWAY_API_KEY` | *(unset)* |

Or export them in the shell before `npm run dev` — same effect, and nothing
touches the disk.

**The key is optional.** The gateway's inbound gate is opt-in: with
`GATEWAY_API_KEY` unset on the fleet, the gate returns before the header is
even read. Leave `LOBES_GATEWAY_API_KEY` unset for such a fleet and the proxy
attaches nothing, correctly.

Neither variable carries the `PUBLIC_` prefix (Astro's client-exposed
namespace) or the `VITE_` prefix (Vite's). That is deliberate and load-bearing:
both toolchains inline prefix-matching variables into the client bundle, so a
key named `PUBLIC_…` or `VITE_…` would be shipped to the browser **by the build
itself**. Do not rename them.

On startup the dev server prints where it will forward and whether a credential
is attached — presence only, never a prefix, a length, or a hash:

```text
[lobes] /v1/* + /capabilities -> http://127.0.0.1:8000; injecting Authorization: Bearer <LOBES_GATEWAY_API_KEY>
```

### Why the Vite dev-server proxy (and not a standalone one)

The plan pinned a tiny standalone `ws` proxy as the fallback, because "Vite ws
upgrade-header injection is unverified on this Vite major" was an open risk.

**It was verified on the installed Vite 8.1.5 and it works**, so the fallback
was not built. Both mechanisms inject correctly — the `headers` option and a
`configure()` hook on `proxyReqWs` — confirmed against a server that refuses
any upgrade without the header. The shipped code uses the `configure()` hook,
because that one can also *remove* headers, and the bundled `http-proxy-3`
pass makes two further guarantees this needs:

- `proxyReqWs` fires after the outgoing headers are assembled and **before**
  the request is flushed, so `setHeader`/`removeHeader` land on the wire.
- both sockets get `setTimeout(0)`, so an **idle** session is never torn down
  by a proxy timeout — a realtime session is silent by design between
  utterances, and a 2-minute idle cull would have been a subtle, intermittent
  disaster.

`ws` and `@types/ws` remain declared in `package.json` (pinned complete before
this wave started) and are now unused. Removing them is a separate, deliberate
change.

The proxy is **dev-server only**, by construction. `astro build` emits static
files and `astro preview` serves them *without* it. A built site opened any
other way cannot reach the gateway at all — which is the intended failure mode
for a local-only tool, not a gap.

## Mounting the connection panel

`src/pages/index.astro` reserves `#connection-mount` /
`[data-mount="connection"]` for the panel, but three wave-2 tasks build inside
`src/` in parallel and all three would collide editing that one file, so the
page is wired by the coordinator after they merge. To wire it:

```astro
---
import ConnectionPanel from "../components/ConnectionPanel.astro";
---
<div class="mount card" id="connection-mount" data-mount="connection">
  <ConnectionPanel />
</div>
```

No props. `src/pages/dev-connection.astro` is the standalone harness — the
smallest page that reproduces a proxy or credential problem with nothing else
on it — and stays useful afterwards.

### Sharing the session with the other islands

One socket serves all three islands. The panel publishes the live connection
**two ways**, because islands built in parallel have no guaranteed load order:

```js
// An island whose script runs BEFORE the panel mounts:
document.addEventListener("lobes:realtime-ready", (event) => {
  const connection = event.detail.connection;
});

// An island whose script runs AFTER:
const connection = window.lobesRealtime;
```

Every notice the panel hears is also re-dispatched on `document` as a
`lobes:realtime` `CustomEvent`, with a `ConnectionNotice` as `detail` — the
zero-import seam for the event log (t12) and mic island (t11):

```js
document.addEventListener("lobes:realtime", (event) => {
  const notice = event.detail;
  // { kind: "state",     state, url, detail }
  // { kind: "event",     event, raw }      ← a parsed server event
  // { kind: "malformed", raw, detail }     ← a text frame that was not JSON
  // { kind: "binary",    byteLength }      ← unexpected on the #151 wire
});
```

The typed module export is the better seam when import order *is* under
someone's control:

```ts
import { mountConnectionPanel } from "../scripts/connection-panel.ts";
const { connection } = mountConnectionPanel(root);
connection.sendEvent({ type: "input_audio_buffer.append", audio: base64 });
```

`connection` is a `RealtimeConnection`: `connect()`, `disconnect()`,
`send(data)`, `sendEvent(obj)`, `subscribe(listener)`, plus `state` and `url`.
`src/scripts/realtime-connection.ts` is DOM-free and takes an injected socket
factory, so it is driven directly in tests.

## What the panel controls

| Control | What it does |
| --- | --- |
| Proxy endpoint | Path (default `/v1/realtime`) resolved against this page's origin — which is what keeps it same-origin and therefore proxied. An absolute `ws://` override is accepted for a proxy on another port. |
| Input sample rate | `24000` (default) or `16000` (skips the server-side resample). Sent as `input_sample_rate`. |
| Server AEC | `none` (default) or `aec`. Leave it at `none`: the browser cancels echo itself via `getUserMedia`. |
| Connect / Disconnect | Opens and closes the one session. The config controls lock while it is live — the bridge fixes the config from the connect URL, so a control that appeared to change it mid-session would be lying. |
| Check gateway | The preflight, below. |

There is **no field for a key**, and there never will be. The browser holds no
credential; a field inviting someone to paste one would put it on the wire as a
query parameter.

Connection state renders as `disconnected` / `connecting` / `open` / `closing`
/ `failed`, each with its own glyph, word and border treatment — never colour
alone (WCAG 1.4.1) — and the region is an `aria-live` status so the transition
is announced.

### Why "Check gateway" exists

**A failed WebSocket handshake looks identical for every cause.** The browser
WebSocket API deliberately hides the HTTP status of a rejected upgrade, so "no
dev server", "gateway down", "401 wrong key" and "404 `role_infeasible` — the
`stt` lane is declared off on this box" all arrive as close code 1006 and
nothing else.

The check does what the socket cannot: `GET /v1/models` through the **same
proxy entry with the same injected credential**, so a 401 there is a 401 on the
session; and `GET /capabilities` (keyless) for whether the `stt` lane is
feasible here at all, naming the `hosted_by` peer when a mesh shape dropped it.

Both go through the proxy for a second reason: the gateway sends no
`Access-Control-Allow-*` headers, so browser HTTP to it is cross-origin-blocked
regardless of authentication.

## Scripts

| Command | What it does |
| --- | --- |
| `npm ci` | Install exactly the lockfile. Node 22+ (`.nvmrc`). |
| `npm run dev` | Dev server **with the proxy** — the only way the site reaches a gateway. |
| `npm run build` | Static build to `dist/`. No proxy in the output. |
| `npm run preview` | Serves `dist/` — **without** the proxy, so no gateway access. |
| `npm run check` | `astro check` — types across `.astro` and `.ts`. |
| `npm test` | `vitest run` — offline fixture tests, no browser, no hardware. |

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| No microphone, no prompt, no error | Not a secure context. The URL is not `localhost` and not HTTPS. See the top of this file. |
| Connection state goes straight to `failed` | Press **Check gateway** — it sees the status the socket cannot. |
| Check says 401 | `LOBES_GATEWAY_API_KEY` does not match the fleet's `GATEWAY_API_KEY`. |
| Check says unreachable | `LOBES_GATEWAY_URL` is wrong, the `ssh -L` forward is down, or the fleet is not up. |
| Check says `stt` lane declared off | This box does not host `stt`; `/v1/realtime` 404s `role_infeasible`. The `hosted_by` peer, when declared, is named in the verdict. |
| Works under `npm run dev`, dead under `npm run preview` | Expected. The proxy is dev-only; there is no gateway route from a built site. |
| Page served over HTTPS, socket refuses | Mixed content: a `wss:` page needs the gateway over TLS or forwarded to localhost. |

## Layout

```text
site/
├── astro.config.mjs              # static output + the dev-server proxy wiring
├── .env.example                  # the proxy's two knobs (copy to .env)
├── proxy/gateway-proxy.mjs       # credential injection, header stripping (pure, tested)
├── src/
│   ├── components/ConnectionPanel.astro   # the panel's markup + scoped styles
│   ├── scripts/realtime-connection.ts     # the socket + state machine (DOM-free)
│   ├── scripts/connection-panel.ts        # the DOM binding + the island seams
│   ├── pages/dev-connection.astro         # standalone harness for the panel
│   └── styles/global.css                  # the ported design system
└── test/                          # offline fixture tests (vitest + jsdom)
```
