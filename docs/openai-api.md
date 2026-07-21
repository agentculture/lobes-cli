# OpenAI-compatible API reference

> Every OpenAI-compatible endpoint lobes serves, and how the gateway routes it.

All endpoints are served on a **single port** (default `:8000`, set by `VLLM_PORT`
in the deployment `.env`). In single-model mode that port is the raw vLLM container
itself; in fleet mode it is the `model-gear-gateway` container, which routes by the
request's `model` field. Clients point at the same URL either way.

## The surface

| Endpoint | Method | Backend | Notes |
|---|---|---|---|
| `/v1/chat/completions` | POST | generate primary (opt-in fallback) | routed by `model` field; unknown/missing → default primary. SSE when `"stream": true`. |
| `/v1/completions` | POST | generate primary (opt-in fallback) | same routing as chat/completions |
| `/v1/embeddings` | POST | `Qwen/Qwen3-Embedding-0.6B` (warm embed gear) | 1024-dim, Matryoshka-truncatable via `"dimensions"` |
| `/v1/rerank` | POST | `Qwen/Qwen3-Reranker-0.6B` (warm rerank gear) | Jina/Cohere shape, sorted best-first |
| `/v1/score` | POST | `Qwen/Qwen3-Reranker-0.6B` (same backend as rerank) | raw cross-encoder scores, input order |
| `/v1/audio/transcriptions` | POST | Parakeet STT via the realtime bridge | multipart `file` upload → `{"text": ...}` |
| `/v1/audio/speech` | POST | Chatterbox TTS via the realtime bridge | text → audio bytes (24 kHz) |
| `/v1/realtime` | GET (WebSocket upgrade) | realtime bridge, tunneled through the gateway | server_vad session; PCM16 mono LE, 24000 Hz default / 16000 Hz accepted (issue #149) — see below |
| `/v1/models` | GET | gateway | OpenAI-standard list of loaded backends (what is hot now) |
| `/v1/models/supported` | GET | gateway | full supported-model catalog (every gear you can switch to; each flagged `loaded`/`default`) |
| `/capabilities` | GET | gateway | the seven-role Colleague contract (`cortex`/`senses`/`muse`/`embedder`/`reranker`/`stt`/`tts`) resolved to live endpoint + metadata — non-OpenAI, lobes-native |
| `/health` | GET | gateway | liveness |

Embeddings, rerank, score, and audio (including the `/v1/realtime` WebSocket
session) require the **fleet overlay** — they are not available when running
the raw vLLM container in single-model mode. The generate endpoints
(`/v1/chat/completions`, `/v1/completions`) and `/v1/models` work in both
modes.

## How routing works

### By name

The gateway inspects the request's `model` field and forwards the request to the
backend that declares that served name (or any alias listed in `GATEWAY_ALIASES`).
The forwarded body's `model` is rewritten to the backend's actual
`--served-model-name` so the backend accepts aliased or defaulted routes without
complaint.

### Default

A missing or unknown `model` value routes to `GATEWAY_DEFAULT_MODEL` (the primary),
so existing single-model clients — the `acp` `vllm-local` provider, plain `curl`
calls — keep working with no changes.

### Failover (generate only)

When an opt-in warm fallback is configured, a chosen generate backend that refuses
the connection or returns a 5xx **before any response body** is retried against the
other generate backend. A 4xx (client error) is returned verbatim — no failover.
Once a 2xx body has started streaming, there is no retry; the client already has
bytes. By default the fleet runs **one** generate backend (the primary), so there
is no failover peer for generate; the embed/rerank gears are separate task families
and are never failover targets for each other.

### Pressure backpressure (busy, `429`)

When the host is under swap/iowait pressure, a full-tier generate request
(`main`/`cortex`, `multimodal`/`senses`, or `muse`) is **shed** with **`429 Too Many
Requests`** rather than silently degraded onto a different model — under
pressure the gateway never substitutes a cheaper or different-capability model
(issue #85). An explicit `model=minor` request is the servable floor and is
always served. The `429` carries:

| Field | Value |
|---|---|
| `Retry-After` | seconds to wait before retrying (`5` by default) |
| `X-Lobes-Tier-Reason` | `busy` |
| Body | `{"error": {"type": "server_busy", "code": "busy", "message": "…"}}` |

This is distinct from a **`502`** (`type: upstream_unavailable`, every backend
down — do *not* retry): a `429` means the model is up but the box is pressured.
**Clients MUST treat `429` + `Retry-After` as a retryable transient and back off**
— the `acp` `vllm-local` provider, colleague, and generic OpenAI SDKs all do.
Send `X-Lobes-Override: true` to force the requested tier and be served instead of
shed (the manual escape hatch). See
[`docs/gateway-fleet.md`](gateway-fleet.md#pressure-policy-and-busy-backpressure)
for the full policy and thresholds.

### Force-strict tool calling (opt-in, colleague#320)

Set `GATEWAY_FORCE_STRICT_TOOLS=1` in the deployment `.env` (unset/false by
default) to have the gateway inject `"strict": true` into every
`tools[i].function` on a `/v1/chat/completions` request that both (a) routes
to the **cortex/primary** lane and (b) carries a non-empty `tools` array —
before forwarding it on. `strict: true` arms vLLM's xgrammar structural-tag
constrained decoding, so a tool call's arguments are grammar-constrained
against the tool's own JSON schema instead of merely parsed after the model
emits them — see
[`docs/qwen3.6-27b-text-nvfp4-mtp.md`](qwen3.6-27b-text-nvfp4-mtp.md) for why
this matters for a thinking model (colleague#320: unconstrained generation
can drift off the tool-call template and get "salvaged" into a mangled call;
the served build's structural-tag call site also needed a request-aware
`reasoning` flag to stay compatible with thinking mode — the paired
`qwen3_coder_thinking` tool-parser plugin, not this knob, supplies that).

- **Caller wins.** A tool that already carries any `strict` key (`true` OR
  `false`) is left untouched — the gateway fills in only an absent key.
  Nothing else about the request changes.
- **Scope.** Only `/v1/chat/completions`, and only when the resolved backend
  is the primary (cortex) lane. `/v1/completions`, embeddings, rerank, and
  audio are never touched; a `multimodal`/`senses` request is unaffected.
  **`muse` is excluded too**, even though it serves tool calls and declares
  `tool_use` — that omission is deliberate, not an oversight: on the muse lane
  the knob is **inert**. Measured live on the 31B (2026-07-17), `strict: true`
  never engages xgrammar there — a schema with a regex xgrammar cannot compile
  is accepted with HTTP 200 rather than failing, no grammar log line is emitted,
  and the output matches `strict: false`. Injecting `strict` would advertise a
  grammar-constrained lane that isn't one. See
  [`docs/gemma-4-31b-nvfp4.md#tool-calling`](gemma-4-31b-nvfp4.md#tool-calling),
  which also records the two *disproven* rationales an earlier draft gave (the
  `supports_required_and_named` flag, which cortex's own parser shares; and an
  EngineCore-crash risk that did not reproduce). The lane set is
  `lobes.gateway.server._STRICT_TOOL_LANES`; widen it only with a live
  transcript showing strict decoding actually constrains decoding there.
- **Retry-without-strict fallback.** If the injected request comes back with
  a 4xx/5xx whose body matches a schema/grammar-compile-failure signature
  (a heuristic substring list — `structural_tag`, `xgrammar`, `grammar`,
  `json_schema` — pending live discovery of the real error text), the
  gateway retries **exactly once** with the original, un-injected body and
  relays that retry's response verbatim (success or failure) — never a
  second retry. The offending tool schema name(s) and an upstream error
  snippet are logged server-side. A failure that does *not* match the
  signature (the caller's own error, or an unrelated owner problem) is
  relayed as-is, with no retry.
- **Default off is byte-identical passthrough.** With the knob unset (or on
  any non-eligible request — no tools, non-primary lane, every tool already
  declaring its own `strict`), none of this logic runs: the request takes
  the exact code path it took before the knob existed.

See [`docs/gateway-fleet.md`](gateway-fleet.md) for where this sits in the
gateway's fleet-fronting behavior, and
[`docs/qwen3.6-27b-text-nvfp4-mtp.md`](qwen3.6-27b-text-nvfp4-mtp.md) for the
tool-parser plugin this pairs with.

### SSE streaming

`"stream": true` requests are relayed chunk-by-chunk with per-chunk flushing.
Normal (non-streaming) responses are buffered with `Content-Length`.

### Audio fan-out

`/v1/audio/*` requests are forwarded by the gateway to the **realtime bridge**
(`model-gear-realtime`, default `http://realtime:8080`, configured via `AUDIO_URL`).
The bridge proxies:

- `POST /v1/audio/transcriptions` → Parakeet STT (`model-gear-stt`, port 9002)
- `POST /v1/audio/speech` → Chatterbox TTS (`model-gear-chatterbox`, port 9000)

The audio overlay is enabled with `lobes init --fleet --audio --apply`. See
[`docs/realtime-pipeline.md`](realtime-pipeline.md) for full bring-up instructions.

### Realtime WebSocket tunnel (issue #149)

`/v1/realtime` reaches the same bridge a different way: not an HTTP forward,
but a **101-upgrade + bidirectional byte tunnel**
(`lobes/gateway/_realtime.py`). The gateway relays the client's WebSocket
handshake to the local `realtime` bridge verbatim, relays the bridge's `101`
back, and then pumps opaque bytes both directions until either side closes —
it never parses the WebSocket protocol itself. A plain `GET` with no
`Upgrade: websocket` header gets **426**; a declared-off `stt` lane gets the
same **404 `role_infeasible`** (naming `hosted_by`) the batch STT route
gets. Unlike the batch audio routes, this tunnel is **never proxied
cross-box** even when `STT_PEER_PROXY` is armed — the #129 proxy-lobes
forwarder is POST-only. See [Realtime session](#realtime-session-v1realtime-websocket)
below for the session contract.

**Auth is opt-in.** Set `GATEWAY_API_KEY` (fallback `CULTURE_VLLM_API_KEY`) in
the deployment `.env` to require `Authorization: Bearer <key>` on every
data-plane request, `/v1/audio/*` included — unset (the default) leaves the
gateway exactly as before, no header inspected. See
[Auth and exposure](#auth-and-exposure) below for the full gated-vs-keyless
route policy and the 401 shape.

## Endpoints in detail

### Chat and completions

Routes to the generate primary (`sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` by
default) or the opt-in fallback. Supply the served model name in `model`, or omit
it to hit the primary.

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP",
    "messages": [{"role": "user", "content": "What is 17 * 23?"}]
  }'
```

Streaming:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP",
    "messages": [{"role": "user", "content": "Count to five."}],
    "stream": true
  }'
```

The `vllm-local` provider in `culture.yaml` points at this endpoint; an unknown
`model` defaults to the primary, so `model: default` also works when you set
`VLLM_SERVED_NAME=default` in `.env`.

### Embeddings

`POST /v1/embeddings` — served by the warm `Qwen/Qwen3-Embedding-0.6B` gear (fleet
only). Native dimension is **1024**; pass `"dimensions"` to Matryoshka-truncate.

Request:

```json
{
  "model": "Qwen/Qwen3-Embedding-0.6B",
  "input": ["text a", "text b"]
}
```

`input` accepts a string or a list of strings. `"dimensions": 512` truncates to any
supported Matryoshka sub-dimension (32 / 64 / 128 / 256 / 512 / 768 / 1024); omit
to get the native 1024-dim output.

Response:

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [/* 1024 floats */]},
    {"object": "embedding", "index": 1, "embedding": [/* 1024 floats */]}
  ],
  "model": "Qwen/Qwen3-Embedding-0.6B",
  "usage": {"prompt_tokens": 4, "total_tokens": 4}
}
```

```bash
curl -s http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-Embedding-0.6B","input":["Hello world"]}'
```

### Rerank

`POST /v1/rerank` — Jina/Cohere-compatible; served by the warm
`Qwen/Qwen3-Reranker-0.6B` gear. Results are sorted **best-first**; `index` is the
position in the original `documents` list.

Request:

```json
{
  "model": "Qwen/Qwen3-Reranker-0.6B",
  "query": "What is the capital of France?",
  "documents": ["Paris is the capital.", "Berlin is the capital.", "Rome is the capital."]
}
```

Response:

```json
{
  "results": [
    {"index": 0, "relevance_score": 0.91},
    {"index": 2, "relevance_score": 0.18},
    {"index": 1, "relevance_score": 0.07}
  ]
}
```

```bash
curl -s http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Reranker-0.6B",
    "query": "capital of France",
    "documents": ["Paris is the capital.","Berlin is the capital."]
  }'
```

### Score

`POST /v1/score` — raw cross-encoder scores from the same
`Qwen/Qwen3-Reranker-0.6B` backend as `/v1/rerank`. Results are in **input order**
(not sorted); use `/v1/rerank` when you need sorted output.

Request:

```json
{
  "model": "Qwen/Qwen3-Reranker-0.6B",
  "text_1": "What is the capital of France?",
  "text_2": ["Paris is the capital.", "Berlin is the capital."]
}
```

`text_1` is the query string; `text_2` is a string or list of strings.

Response:

```json
{
  "object": "list",
  "data": [
    {"index": 0, "score": 0.91},
    {"index": 1, "score": 0.07}
  ]
}
```

```bash
curl -s http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Reranker-0.6B",
    "text_1": "capital of France",
    "text_2": ["Paris is the capital.","Berlin is the capital."]
  }'
```

### Audio transcriptions (STT)

`POST /v1/audio/transcriptions` — multipart upload; served by Parakeet NeMo ASR
via the realtime bridge. Requires the `--audio` fleet overlay.

```bash
curl -s http://localhost:8000/v1/audio/transcriptions \
  -F "file=@clip.wav" \
  -F "language=en"
```

The backend model is fixed (`nvidia/parakeet-tdt-0.6b-v2`), so no `model` field is
needed; `language` is accepted for forward-compatibility (Parakeet is English-only).

Response:

```json
{"text": "the transcribed words here"}
```

### Audio speech (TTS)

`POST /v1/audio/speech` — served by Chatterbox TTS via the realtime bridge.
Returns **audio bytes** (24 kHz). Requires the `--audio` fleet overlay.

```bash
curl -s http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"chatterbox","input":"Hello from lobes.","voice":""}' \
  -o speech.wav
```

The response body is raw audio — pipe to a file or an audio player. Leave `voice`
empty (or null) for Chatterbox's built-in default voice; set it to a `.wav` path on
the sidecar for zero-shot voice cloning (passed through as `audio_prompt_path`). See
[`docs/chatterbox-tts.md`](chatterbox-tts.md) for the voice-cloning contract.

Audio endpoints are gated by the same opt-in bearer check as every other POST
route — see [Auth and exposure](#auth-and-exposure) below.

### Realtime session (`/v1/realtime` WebSocket)

`GET /v1/realtime` with an `Upgrade: websocket` handshake — a persistent
`server_vad` session that replaces separate STT batch calls with one
connection: stream PCM audio in, receive VAD boundary and transcription
events back on the same socket. Requires the `--audio` fleet overlay
(issue #149).

**Wire format: PCM16 mono little-endian**, streamed as binary WebSocket
frames. `input_sample_rate` defaults to **24000 Hz**; **16000 Hz is also
accepted** — the server resamples 24 kHz down to 16 kHz itself before
feeding Silero VAD and Parakeet (both native at 16 kHz), so a client never
needs to know or match that rate. Any other rate is rejected as an invalid
session config before any audio is accepted.

Session config is set via **connect-URL query parameters**, not a first
message:

```text
ws://localhost:8000/v1/realtime?input_sample_rate=16000
```

| Param | Default | Accepted |
|---|---|---|
| `input_audio_format` | `pcm16` | `pcm16` only |
| `input_sample_rate` | `24000` | `24000` or `16000` |
| `input_channels` | `1` | `1` (mono) only |
| `turn_detection` | `server_vad` | `server_vad` only |
| `aec_mode` | `none` | `none` or `aec` |

```python
import websocket  # pip install websocket-client

ws = websocket.create_connection("ws://localhost:8000/v1/realtime?input_sample_rate=16000")
print(ws.recv())           # {"type": "session.created", ...} — confirms negotiated config
ws.send_binary(pcm_chunk)  # PCM16 mono LE; any chunk size, the server reassembles the stream
print(ws.recv())           # speech_started / speech_stopped / transcription.completed / error
```

Events come back as JSON text frames:

- `session.created` — sent immediately after the handshake; confirms the
  negotiated config (read effective defaults off it if you sent none).
- `input_audio_buffer.speech_started` / `...speech_stopped` — server-side
  Silero VAD boundaries. A stopped turn carries `reason: "silence"` (normal)
  or `reason: "max_turn"` — the max-turn cap (`VAD_MAX_TURN_MS`, default
  30000 ms) force-committing an unusually long, never-silent turn.
  **`max_turn` is not an error** — no `error` event fires on this path.
- `conversation.item.input_audio_transcription.completed` — the committed
  turn's Parakeet transcript, on the same connection.
- `error` — a named `code`: `invalid_session_config` (rejected before any
  session exists), `vad_unavailable` (Silero down — distinct from ordinary
  silence, which emits nothing), `stt_forward_failed` (the Parakeet forward
  failed; a turn is never silently dropped).

**Sessions are ephemeral.** There is no resume: a disconnect for any reason
(idle, mid-speech, mid-transcription) tears the session down completely, and
a reconnecting client gets a brand-new session id — reconnect-and-restart is
the client contract, not a bug to work around.

Reached the same way as the batch audio routes — through the gateway,
tunneled to the realtime bridge (101 upgrade + byte relay, see
[Realtime WebSocket tunnel](#realtime-websocket-tunnel-issue-149) above) —
gated by the same opt-in bearer check (a missing/wrong key is rejected
before any tunnel is opened); see [Auth and exposure](#auth-and-exposure)
below.

**Live status: DECLARED/UNVALIDATED.** The session/VAD logic is proven by
the offline unit suite with a scripted fake VAD; nothing here has been
exercised on real hardware yet — no `docs/evidence/` transcript exists for
issue #149. See
[`docs/realtime-pipeline.md`](realtime-pipeline.md#the-v1realtime-websocket-session-issue-149)
for the full contract, the ephemeral-session restart contract in detail, and
the #149 baseline this redeems.

### Model list (loaded)

`GET /v1/models` — OpenAI-standard list of backends currently loaded in GPU memory.
In single-model mode this is one entry; in fleet mode it includes the generate
primary plus the embedding and reranker gears.

```bash
curl -s http://localhost:8000/v1/models
```

### Supported catalog

`GET /v1/models/supported` — the full lobes supported catalog: every gear you
can switch to, each flagged `loaded` (in GPU memory now) and `default` (the primary
the gateway defaults to). This is the HTTP equivalent of `lobes overview --list`.

```bash
curl -s http://localhost:8000/v1/models/supported
```

### Capabilities (the seven-role Colleague contract)

`GET /capabilities` — the SEVEN first-class, Colleague-facing roles (`cortex`,
`senses`, `muse`, `embedder`, `reranker`, `stt`, `tts` — issue #81), each
resolved to
live metadata: `role`, `model`, `runtime`, `endpoint`, `path`, `context`,
`quant`, `mtp`, `responsibilities`, `forbidden_responsibilities`, `ready`, and
`loaded`. This is the discovery contract a Colleague client uses to drive the
fleet by capability instead of a hardcoded model id — `lobes capabilities
--json` returns the identical shape over the CLI. Non-OpenAI (lobes-native), a
sibling to `/status`.

Generate requests are addressed by capability-tier alias — `model=main|minor|
multimodal|muse` (back-compat `hard|cheap|normal`), or the Colleague-role
names `model=cortex|senses|muse`. `muse` (the opt-in creative/ideation lobe,
Gemma 4 31B NVFP4) is hosted only by a muse-hosting deployment shape: on a
deployment that doesn't host it, `model=muse` gets an honest `404
role_infeasible` (never a silent fallback to the primary — the inverted
feasibility default; see
[`docs/gateway-fleet.md`](gateway-fleet.md#generate-lane-tier-aliases)).

```bash
curl -s http://localhost:8000/capabilities
```

```json
{
  "cortex": {
    "role": "cortex", "model": "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP",
    "runtime": "vllm", "endpoint": "http://localhost:8000",
    "path": "/v1/chat/completions", "context": 131072, "quant": "modelopt",
    "mtp": true, "responsibilities": ["reasoning", "deciding", "..."],
    "forbidden_responsibilities": [], "ready": true, "loaded": true
  },
  "senses": { "...": "..." },
  "muse": { "...": "..." },
  "embedder": { "...": "..." },
  "reranker": { "...": "..." },
  "stt": { "...": "..." },
  "tts": { "...": "..." }
}
```

**All seven roles** report this **one** client-reachable gateway `endpoint` —
including `stt`/`tts` — because routing is by the `model` field / OpenAI `path`,
not by distinct URLs (issue #87). The gateway advertises the origin you dialed
(the request `Host` header; override with `GATEWAY_PUBLIC_URL` for a tunnel), so
`endpoint` is reachable as-is and internal hosts are never leaked. For `stt`/`tts`,
`GET /capabilities` reports a **live** readiness probe (issue #89) — `ready: true`
only when an audio round-trip would truly succeed. See
[`docs/colleague-stack.md`](colleague-stack.md) for the full contract
(responsibilities per role, the cortex/senses↔primary/multimodal mapping,
`lobes up <role>`, `lobes measure`, and the client-flow / rename-safety proof).

## Loaded vs. supported

Two questions that look alike but are not:

| Question | CLI | HTTP |
|---|---|---|
| What *can* I run? (catalog) | `lobes overview --list` | `GET /v1/models/supported` |
| What's *loaded* right now? | `lobes fleet status` | `GET /v1/models` |
| What's the deployment *set* to serve? | `lobes status` / `lobes whoami` | — |

Mnemonic: the catalog is what's on the **menu** (and which dishes have been
cooked); `GET /v1/models` is what's **hot now**.

`lobes status` / `lobes whoami` report the model the deployment is configured to
serve (from `.env`) plus container health — normally the same model, but it is
configuration, not a live query. For runtime truth, query `/v1/models`.

See [`docs/gateway-fleet.md`](gateway-fleet.md#supported-catalog-vs-warm-backends)
for the full discussion.

## Auth and exposure

### Single-model deployment

Set `CULTURE_VLLM_API_KEY` in `$HOME/.lobes/.env` before serving. vLLM then
requires `Authorization: Bearer $CULTURE_VLLM_API_KEY` on every request. An empty
key leaves the API open — only safe for local development.

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $CULTURE_VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"hi"}]}'
```

Generate the key with `python3 scripts/gen-api-key.py` (writes to the deployment
`.env`; `--show` to print, `--force` to rotate), then `lobes serve --apply` to
enforce it.

### Fleet gateway

Set `GATEWAY_API_KEY` in the fleet `.env` (fallback `CULTURE_VLLM_API_KEY` —
the first non-blank of the two wins) to require `Authorization: Bearer <key>`
on the gateway's own data-plane routes, `/v1/audio/*` included. Unset — the
default — leaves the gateway exactly as before: no header is ever inspected.

**Gated:** every `POST` (chat/completions, completions, embeddings, rerank,
score, audio) and `GET /v1/models` + `GET /v1/models/supported`. **Keyless by
design**, regardless of the key: `GET /health` (container/peer probes must
reach it before any key is distributed), `GET /capabilities` (the
control-plane discovery surface peers read to learn what a box hosts, before
they hold a key), and `GET /status` (operator observability, no inference).

A missing/wrong/malformed key gets `401` with an OpenAI-shaped
`invalid_api_key` body and a `WWW-Authenticate: Bearer` header:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"cortex","messages":[{"role":"user","content":"hi"}]}'
```

```json
{"error": {"message": "Invalid API key. Pass this gateway's configured key as 'Authorization: Bearer <key>'.", "type": "invalid_api_key", "code": "invalid_api_key"}}
```

The message never echoes what was sent nor any part of the expected key; the
comparison is timing-safe (`hmac.compare_digest`). Keeping the gateway port
off the public internet — the Cloudflare Tunnel (`lobes tunnel`), Cloudflare
Access, an IP allowlist, or binding to localhost and exposing only via the
tunnel — remains good practice, but with `GATEWAY_API_KEY` set it is now
**defense-in-depth on top of** the bearer gate, not the only protection.

**A proxied response carries an extra header.** When this gateway forwards a
request to a peer box for a role it dropped (proxy-lobes, opt-in — see
[`docs/gateway-fleet.md#proxy-lobes-the-third-lobe-state-opt-in`](gateway-fleet.md#proxy-lobes-the-third-lobe-state-opt-in)),
the response carries `X-Lobes-Proxied-By: <peer origin>`; a locally-served
response never does. `GET /v1/models` only lists a proxied role's id while a
live probe of the peer's own `/v1/models` confirms it actually serves that
id — the same "advertised implies reachable" rule (issue #92) a local backend
is already held to, extended across the box boundary.

### Public exposure via Cloudflare Tunnel

`lobes tunnel --apply` publishes the local API at an owner-chosen hostname through
a Cloudflare Tunnel — no inbound ports, no static IP required. Set
`CULTURE_VLLM_API_KEY` before tunnelling.

```bash
lobes tunnel       # dry-run: prints the cloudflared command + public URL
lobes tunnel --apply   # start the tunnel in the background
lobes tunnel --stop --apply   # tear it down
```

See the README "Expose the API" section and `lobes explain tunnel` for the full
two-step provisioning flow (`cultureflare` + `lobes tunnel`).

## See also

- `lobes explain gateway` — routing semantics (name / default / failover / SSE)
- `lobes explain fleet` — the multi-container fleet topology
- `lobes explain roles` — the seven-role Colleague contract (`GET /capabilities`)
- `lobes explain embeddings` — `/v1/embeddings` request/response detail
- `lobes explain rerank` — `/v1/rerank` request/response detail
- `lobes explain score` — `/v1/score` request/response detail
- `lobes explain tunnel` — Cloudflare Tunnel bring-up
- `lobes explain realtime` — the `/v1/realtime` session surface, in-CLI
- [`docs/gateway-fleet.md`](gateway-fleet.md) — full fleet topology, memory guidance, live validation findings
- [`docs/colleague-stack.md`](colleague-stack.md) — the seven-role Colleague contract, `GET /capabilities` JSON shape, `lobes up`/`measure`/`benchmark --profile`
- [`docs/realtime-pipeline.md`](realtime-pipeline.md) — audio overlay bring-up (STT + TTS), the `/v1/realtime` session contract, health/readiness, runbooks
- [`docs/chatterbox-tts.md`](chatterbox-tts.md) — Chatterbox TTS details, voice prompting
