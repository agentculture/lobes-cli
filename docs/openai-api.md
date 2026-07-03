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
| `/v1/models` | GET | gateway | OpenAI-standard list of loaded backends (what is hot now) |
| `/v1/models/supported` | GET | gateway | full supported-model catalog (every gear you can switch to; each flagged `loaded`/`default`) |
| `/capabilities` | GET | gateway | the six-role Colleague contract (`cortex`/`senses`/`embedder`/`reranker`/`stt`/`tts`) resolved to live endpoint + metadata — non-OpenAI, lobes-native |
| `/health` | GET | gateway | liveness |

Embeddings, rerank, score, and audio require the **fleet overlay** — they are not
available when running the raw vLLM container in single-model mode. The generate
endpoints (`/v1/chat/completions`, `/v1/completions`) and `/v1/models` work in both
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

**Known limitation:** the fleet gateway is not auth-aware. `CULTURE_VLLM_API_KEY`
is enforced by vLLM on the single-model serve path, but the gateway is a
pass-through — the bearer token does not extend to any of its proxied endpoints,
`/v1/audio/*` included. Per-endpoint gateway auth is planned for a later release.

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

**Known limitation:** the gateway does not yet extend the `CULTURE_VLLM_API_KEY`
bearer token to audio endpoints. Plan: per-endpoint auth in a later release.

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

### Capabilities (the six-role Colleague contract)

`GET /capabilities` — the SIX first-class, Colleague-facing roles (`cortex`,
`senses`, `embedder`, `reranker`, `stt`, `tts` — issue #81), each resolved to
live metadata: `role`, `model`, `runtime`, `endpoint`, `path`, `context`,
`quant`, `mtp`, `responsibilities`, `forbidden_responsibilities`, `ready`, and
`loaded`. This is the discovery contract a Colleague client uses to drive the
fleet by capability instead of a hardcoded model id — `lobes capabilities
--json` returns the identical shape over the CLI. Non-OpenAI (lobes-native), a
sibling to `/status`.

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
  "embedder": { "...": "..." },
  "reranker": { "...": "..." },
  "stt": { "...": "..." },
  "tts": { "...": "..." }
}
```

All four gateway-fronted roles (`cortex`/`senses`/`embedder`/`reranker`) share
this **one** gateway `endpoint` — routing between them is by the `model`
field the contract hands back, not by distinct URLs. See
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

The fleet gateway is **not yet auth-aware** — the bearer token does not yet extend
to the gateway's proxied endpoints (including `/v1/audio/*`). Per-endpoint auth is
planned. In the meantime, keep the gateway port off the public internet: use the
Cloudflare Tunnel (`lobes tunnel`) on the single-model deployment, or bind the
fleet gateway to localhost and expose only via the tunnel.

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
- `lobes explain roles` — the six-role Colleague contract (`GET /capabilities`)
- `lobes explain embeddings` — `/v1/embeddings` request/response detail
- `lobes explain rerank` — `/v1/rerank` request/response detail
- `lobes explain score` — `/v1/score` request/response detail
- `lobes explain tunnel` — Cloudflare Tunnel bring-up
- [`docs/gateway-fleet.md`](gateway-fleet.md) — full fleet topology, memory guidance, live validation findings
- [`docs/colleague-stack.md`](colleague-stack.md) — the six-role Colleague contract, `GET /capabilities` JSON shape, `lobes up`/`measure`/`benchmark --profile`
- [`docs/realtime-pipeline.md`](realtime-pipeline.md) — audio overlay bring-up (STT + TTS), health/readiness, runbooks
- [`docs/chatterbox-tts.md`](chatterbox-tts.md) — Chatterbox TTS details, voice prompting
