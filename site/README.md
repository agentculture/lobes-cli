# lobes realtime test harness

An Astro site, run locally, for driving the fleet's `GET /v1/realtime` WebSocket
session from a browser: mic in, live event stream, audio out. It exists so the
realtime surface can be *experienced* — VAD boundaries, transcripts,
interruptions scrolling past in real time — instead of inferred from terminal
prints.

**It is never deployed — it runs locally.** There is no adapter, no `site:`
URL, and no workflow under `.github/workflows` that publishes it — and none
should be added. The only CI job that touches this directory builds it, so a
broken site fails a PR; nothing ships it anywhere. Issue #151 records that as a
scope boundary, not an oversight. An operator *may* make the local `npm run
dev` reachable from anywhere by fronting it with a tunnel and an SSO gate — see
[Reaching it from anywhere](#the-alternative-a-tunnel-behind-sso). That is the
same local process, not a deployment.

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

There are three ways out. Use the first one unless you need another.

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

### The alternative: a tunnel behind SSO

To use the harness from any device without forwarding ports (a phone, say),
leave `npm run dev` running on the box and front it with a Cloudflare Tunnel
whose hostname is gated by **Cloudflare Access** for the operator's identity
alone. The page is then served over HTTPS, so it is a secure context and the
microphone works. Its WebSocket dials `wss:` on the same origin, so the
credential-injecting proxy (below) still holds the key, and the browser never
receives it.

Two settings, both in untracked places:

- In `.env`, set `LOBES_SITE_ALLOWED_HOSTS` to the tunnel's public hostname.
  Vite rejects any `Host` header outside its loopback default with "Blocked
  request", and a tunnel request carries the public name.
- Set up the tunnel's route, DNS record and Access app+policy in Cloudflare,
  never in this repo. This repo names no deployment's hostname and no
  operator's email.

**The Access gate is the only thing standing between the internet and the
gateway.** Setting a gateway key doesn't change that for tunnel users,
because this proxy adds the key to every request that reaches it. Anyone
past Access is the operator as far as the gateway can tell, whether or not
`GATEWAY_API_KEY` is set. Never front the dev server with a tunnel that has
no Access app on it.

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
for a locally run tool, not a gap.

**The connect query string (`input_sample_rate`, `aec_mode`, `language`, …)
rides through unchanged.** `buildProxyConfig` sets no `rewrite`,
`pathRewrite`, or `ignorePath` option, so `http-proxy` forwards the
incoming request's full `url` — path *and* query string — to the gateway
verbatim; nothing here parses or copies the query string by hand, and there
is nothing to keep in sync when a new connect param (like `language`, added
for hebrew-realtime) shows up. `test/gateway-proxy.test.ts` asserts those
three options stay unset for exactly this reason. This was verified by
reading the code path, not by a live browser round trip in this sandbox —
see "What I could not verify" below.

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
| Language | Free text, defaulting to `he` (Hebrew) — **this harness's own default**, not the server's `"en"`. Sent as the connect URL's `language` param. Any short code is accepted here (`parse_language` server-side is a shape check, not a registry); a datalist offers `he`/`en` as a convenience. |
| Conversation (talks back) | Off by default (ears-only). On: sends `response.create` right after connecting, so every committed turn gets a spoken reply. |
| Declare demo tools | Off by default. On: declares this harness's four demo tools via `session.update` the moment `session.created` arrives, and only arms (`response.create`) once `session.updated` echoes the declaration back — see "Tools", below. Requires Conversation to be on too. |
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

## Hebrew — language, RTL, tools, and latency

The fleet's realtime stack now speaks Hebrew end to end (ivrit.ai Whisper
STT, Gemma 4 26B, Chatterbox Multilingual Hebrew TTS, Silero VAD, tool
calling with OpenAI Realtime event names — see `docs/specs/2026-09-18-
hebrew-realtime.md` and `scripts/realtime-he-accept.py`, the reference
client this harness's tool flow mirrors). This site follows:

### Language

The connection panel's **Language** field defaults to `he` and is sent as
the connect URL's `language` query param (`parse_session_config`'s
`language` key, `lobes/realtime/_session.py`). It is free text, not a
closed picker — a datalist offers `he`/`en` for convenience, but any short
code the server accepts (`parse_language` is a SHAPE check: "a 2-3 letter
primary subtag, optionally with a region/script subtag" — `he`, `en`,
`pt-BR`, …) can be typed. Leave it blank to omit the param entirely and
take the server's own default (`"en"`).

### RTL rendering

Every text node holding a transcript or a reply — in the raw event log's
per-row detail line, and in the chat-style Conversation view below — carries
`dir="auto"`. The browser's Unicode Bidi Algorithm then picks left-to-right
or right-to-left **per element**, from that element's own first strong
character: a Hebrew turn renders RTL, an English one is untouched, and a
mixed Hebrew+Latin turn (a Hebrew sentence naming an English tool argument,
say) does not break the surrounding page layout, because the direction
choice is scoped to that one text node, never the whole page.

The Conversation view additionally declares a font stack
(`--font-rtl-safe` in `ConversationView.astro`) that actually has Hebrew
glyphs: the page's own display/body fonts (Fraunces, Albert Sans) are
self-hosted **Latin-only** subsets (see `Layout.astro`'s `woff2` imports),
so a Hebrew character in either has no glyph there — the browser already
falls through per character to whatever comes next in the CSS font stack.
`--font-rtl-safe` makes that "next" explicit and Hebrew-first
(`"Noto Sans Hebrew", "Arial Hebrew", "Segoe UI", Tahoma, Arial, …`) rather
than relying on an accidentally-adequate system fallback.

### Conversation view

`src/components/ConversationView.astro` (built by
`src/scripts/conversation-view.ts`) renders a chat-style transcript beside
the raw event log: what was **heard** (a transcript), what was **said** (a
reply), a **tool** row (the tool name, its arguments, and — once this
browser sends the answer — its result), and an **interrupted** marker on a
barge-in. It listens on the exact same `window` `"lobes:realtime-event"`
seam `EventStream.astro` already uses, plus one extra: `connection-
panel.ts` dispatches `window` `"lobes:tool-result"`
(`TOOL_RESULT_EVENT`) the moment it sends a tool's `function_call_output`,
because the server never echoes a client's own tool answer back on the
wire — that is the one thing the Conversation view cannot learn purely by
watching inbound events.

### Tools

The **Declare demo tools** checkbox on the connection panel offers four
small, pure, in-browser tools (`src/scripts/demo-tools.ts`, no network):

| Tool | What it does |
| --- | --- |
| `get_current_time` | Returns the current local ISO timestamp and the weekday name in Hebrew. |
| `roll_dice` | Rolls one die (`sides`, default 6, 2-1000). |
| `remember_note` | Remembers a short text note for the rest of the session (in-memory, cleared on reconnect). |
| `list_notes` | Lists every note remembered so far this session. |

**How arming works, step by step** (mirrors `scripts/realtime-he-accept.py`'s
reference shape, and keeps exactly ONE arming path —
`connection-panel.ts`'s `maybeArm`, the single place this file ever sends
`response.create` to start a session):

1. Press Connect with both **Conversation** and **Declare demo tools**
   checked.
2. The socket opens. Arming does **not** fire yet — `maybeArm()` sees tools
   are declared but not yet ready, and waits.
3. `session.created` arrives. The panel sends `session.update` with the four
   tools (OpenAI Realtime FLAT shape: `{type: "function", name,
   description, parameters}`) and `tool_choice: "auto"`.
4. `session.updated` echoes the declaration back. The panel calls
   `maybeArm()` again — this time it sends `response.create`, the ONE
   arming send for this session.
5. On a committed turn, the model may call a tool:
   `response.function_call_arguments.done` arrives with a `call_id`, a
   `name`, and `arguments` (a JSON **string**). The panel parses it (never
   throwing — malformed JSON becomes a `{"error": "..."}` output string,
   exactly like a real tool reporting its own failure), runs the tool, and
   answers with `conversation.item.create` carrying a
   `function_call_output` item, then a **second**, independent
   `response.create` to continue the SAME response (not a new arming — the
   session is already armed). The Conversation view's tool row fills in the
   result once this send goes out.

With **Declare demo tools** off, arming is unchanged from its original
shape: `response.create` fires the moment the socket reaches `open`.

Malformed tool-call arguments, an unknown call id, and every other tool
failure mode never throw — see `demo-tools.ts`'s `executeDemoTool` and its
test suite. Server-side, a tool call that never gets an answer (a barge-in
before the browser could respond) leaves that call **closed**
(`lobes/realtime/_conversation.py`'s `TOOL_OUTPUT_CALL_CLOSED`); a late
answer to it is refused as a named `invalid_wire_event` with
`call_closed` in the message — the event log renders that error like any
other named error, and the Conversation view's tool row is left showing
"no answer seen" (see `event-fixtures.ts`'s interrupted-tool-turn
fixture for a worked example of the shape).

### Latency table

The Conversation view's latency table is fed **only** by
`response.done`'s optional `timings` mapping (`stt`/`generate`/
`tool_wait`/`phonikud`/`tts`/`first_delta`, all milliseconds) — an
unmeasured stage is shown as `—`, never invented as `0`. Rows are newest
first; the `first_delta` column is highlighted, and a running median of
`first_delta` across the session is shown above the table. A
`response.done` with no `timings` at all adds no row. An unknown extra
timing key (a future streaming change might add `first_sentence` /
`first_audio_ready`) is tolerated and carried through rather than dropped.

### What a healthy Hebrew tool turn looks like

- **Conversation view:** a "You" row with Hebrew text rendering
  right-to-left, a "tool · get_current_time" row showing its JSON
  arguments and (moments later) its JSON result, then a "lobes" row with
  the spoken reply — also RTL.
- **Latency table:** a new row at the top with `stt`, `generate`,
  `tool_wait`, `phonikud` and `tts` all populated (Hebrew replies exercise
  the niqqud-restoration `phonikud` stage that an English deployment never
  reports) and a `first_delta` figure.
- **Event log:** `session.updated` right after `session.created` (echoing
  the four tool names), then the usual boundary/transcription/response
  sequence with one `response.function_call_arguments.done` row in the
  middle.

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

## What could not be verified in this environment

This work was done in a sandbox with no browser and no reachable fleet
gateway. Everything above the fixture/unit-test level — a live Hebrew
session with real STT/generate/TTS, a real `phonikud` stage, a real tool
round trip against the deployed Gemma 4 26B, real RTL rendering in an
actual browser, the `ssh -L` forward, and the query-string passthrough
through a REAL Vite dev server proxying to a REAL gateway — is unverified
here and needs a live pass by the operator following the steps above.
`npm test` / `npm run check` / `npm run build` (below) are what this
environment could run.

## Known limitation: echo-cancelled microphone required for barge-in

Barge-in — speaking over a reply to interrupt it — depends on the server
never hearing its own synthesized voice as if it were the operator
speaking. This site never mutes the microphone to fake that (see
`no-mic-mute.test.ts` and `mic-capture.ts`'s own module doc); instead it
relies entirely on the **browser's own echo cancellation**:
`mic-capture.ts`'s `MIC_AUDIO_CONSTRAINTS` requests
`echoCancellation: true` on the `getUserMedia` track. That constraint is
what makes an always-open mic and a barge-in-capable session possible at
all in a browser tab with no server-side AEC — but it is a **request**, not
a guarantee: some devices, drivers, or browsers honour it poorly or not at
all, especially over a laptop's built-in speakers at high volume, or a
Bluetooth headset with its own (sometimes conflicting) echo path. A wired
headset, or a device with real hardware AEC, gives the most reliable
result. If barge-in does not seem to interrupt cleanly, suspect the
device's echo cancellation before suspecting the server.

## Layout

```text
site/
├── astro.config.mjs              # static output + the dev-server proxy wiring
├── .env.example                  # the proxy's two knobs (copy to .env)
├── proxy/gateway-proxy.mjs       # credential injection, header stripping (pure, tested)
├── src/
│   ├── components/ConnectionPanel.astro   # the panel's markup + scoped styles
│   ├── components/ConversationView.astro  # chat-style transcript + latency table
│   ├── scripts/realtime-connection.ts     # the socket + state machine (DOM-free)
│   ├── scripts/connection-panel.ts        # the DOM binding + tools/language/arming
│   ├── scripts/conversation-view.ts       # the transcript/latency rendering model
│   ├── scripts/demo-tools.ts              # the four demo tools (pure, no network)
│   ├── pages/dev-connection.astro         # standalone harness for the panel
│   └── styles/global.css                  # the ported design system
└── test/                          # offline fixture tests (vitest + jsdom)
```
