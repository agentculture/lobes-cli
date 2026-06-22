# Fleet: the Qwen primary + embedding/reranker gears behind one OpenAI gateway

The **fleet** runs the always-warm Qwen generate primary — plus two tiny
co-resident **embedding** and **reranker** gears — behind a single stdlib
OpenAI-compatible gateway, managed by lobes as four Docker containers. It is
an alternative to the bare single-model deployment — scaffold it with
`lobes init --fleet` (the single-model `lobes init` is unchanged and remains the
default). The gateway routes by **task family** (generate / embed / score /
rerank); there is **one generate backend by default** and a warm *generate*
fallback is opt-in (see "Adding a fallback").

## Why

The single-model deployment serves one model on `:8000` and `lobes switch` swaps
it (freeing the prior model). The fleet instead puts a stable OpenAI endpoint in
front of the primary, so:

- existing clients (the acp `vllm-local` provider, `curl`, …) point at `:8000`
  and keep working — an unknown/missing `model` defaults to the primary;
- the gateway can route additional models by name and fail over **if** a second
  backend is wired up;
- the same front fans `/v1/audio/*` out to the audio overlay (`--audio`).

On the DGX Spark (GB10, 128 GB unified memory) the primary — a hybrid-Mamba
**27B** — runs solo at its load-tested headroom (util 0.6, full 256K context,
~75 GiB), owning the box. The prior co-resident dense **24B** Mistral fallback was
removed (two ~30B NVFP4 models do not co-fit a shared GB10 — see "Live validation
findings"); Mistral stays a selectable catalog candidate (`lobes overview --list`)
and the opt-in fallback example.

## Topology

```text
client / acp ──:8000──▶ model-gear-gateway   (python -m lobes.gateway)
                          │  route by `model` / task family
                          ├──▶ model-gear-vllm-primary  :8000  generate (→ failover if a fallback is wired)
                          ├──▶ model-gear-vllm-embed     :8000  embed (/v1/embeddings)
                          └──▶ model-gear-vllm-rerank    :8000  score/rerank (/v1/rerank, /v1/score)
```

Four containers by default, all `restart: unless-stopped`:

| Container | Role | Host port |
|---|---|---|
| `model-gear-gateway` | stdlib reverse proxy (the single OpenAI front) | `${VLLM_PORT:-8000}` |
| `model-gear-vllm-primary` | generate primary (default: `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`) | internal only |
| `model-gear-vllm-embed` | embedding gear (`Qwen/Qwen3-Embedding-0.6B`, `/v1/embeddings`) | internal only |
| `model-gear-vllm-rerank` | reranker gear (`Qwen/Qwen3-Reranker-0.6B`, `/v1/rerank` + `/v1/score`) | internal only |

The backends are reachable only on the compose network (`http://vllm-primary:8000`,
`vllm-embed:8000`, `vllm-rerank:8000`); only the gateway is published to the host. The
gateway needs no Docker socket access — compose owns the lifecycle; the gateway
only routes.

Each vLLM gear runs through `mg-logwrap` so its output (and any crash trace)
persists to per-boot files under the host log dir and **survives restart/recreate** —
read them with `lobes logs {primary,embed,rerank}` even after a container is gone.
See [docs/durable-logs.md](durable-logs.md) (issue #50).

### Adding a fallback

The gateway adds a second backend **only** when `FALLBACK_URL` or
`FALLBACK_SERVED_NAME` is set. To add a warm fallback: define a `vllm-fallback`
service in the fleet compose (mirror `vllm-primary` with the fallback's model /
quantization / tokenizer / tool-parser), add it to the gateway's `depends_on`,
set `FALLBACK_URL` + `FALLBACK_SERVED_NAME` on the gateway, and **drop both
`*_GPU_MEM_UTIL` values** so they sum well under 1.0. The archived dense Mistral
fallback config is in git history and
[`docs/mistral-small-3.2-24b-nvfp4.md`](mistral-small-3.2-24b-nvfp4.md).

## The gateway

A pure-stdlib (`http.server` + `http.client`, no third-party deps) reverse proxy:

- **Name routing** — a request's `model` routes to the backend that serves it,
  plus any `GATEWAY_ALIASES`. The forwarded body's `model` is rewritten to the
  backend's `--served-model-name` so the backend accepts aliased/default routes.
- **Default model** — a missing or unknown `model` routes to
  `GATEWAY_DEFAULT_MODEL` (the primary).
- **Failover** — when a fallback is wired up, a chosen backend that refuses the
  connection or returns a 5xx **before any response body** is retried against the
  other backend. (One generate backend by default → no generate peer to fail over
  to; the embed/rerank gears are separate task families, not failover targets.) A 4xx is a
  client error (returned verbatim, no failover). Once a 2xx body starts streaming
  there is no retry — the client already has bytes.
- **Streaming** — `"stream": true` (SSE) is relayed chunk-by-chunk with per-chunk
  flushing; normal JSON is buffered with `Content-Length`.
- **Endpoints** — `/v1/chat/completions`, `/v1/completions` (generate primary),
  `/v1/embeddings` (the embedding gear), `/v1/rerank` + `/v1/score` (the reranker
  gear), `/v1/audio/transcriptions` + `/v1/audio/speech` (the `--audio` overlay
  only — fanned to the realtime bridge → Parakeet STT / Chatterbox TTS),
  `/v1/models` (OpenAI-standard, lists the loaded backend(s)),
  `/v1/models/supported` (the full supported-model catalog — every gear you can
  change to, each flagged `loaded` / `default`), `/health` (gateway liveness), and
  `/status` (the live fleet aggregate — see below).
  See [Supported catalog vs. warm backends](#supported-catalog-vs-warm-backends)
  for what `/v1/models` and `/v1/models/supported` each mean.
- **`GET /status`** — a lobes-native (non-OpenAI) JSON aggregate the gateway
  fans out to each backend's `/health` + `/metrics` and returns as
  `{object: "lobes.fleet_status", default_model, busy: {running, waiting},
  backends: [{name, task, served_name, health, metrics}], endpoints}`. The backends
  are internal-only, so the gateway is the only thing that can see them — this is
  the source for `lobes overview --live`.

The gateway image is built from the scaffolded `Dockerfile.gateway`
(`pip install lobes-cli==${MODEL_GEAR_VERSION}`, as a non-root user); `lobes init
--fleet` pins `MODEL_GEAR_VERSION` to the running lobes-cli release. The version
is required (pinning keeps the image reproducible); from-source/dev boxes that run
ahead of a PyPI release point `MODEL_GEAR_VERSION` at a published TestPyPI `.devN`
build.

### Auth (known limitation)

The gateway is a **pass-through** and is **not auth-aware** — it does not inspect
or validate `Authorization` headers. `CULTURE_VLLM_API_KEY` is enforced by vLLM on
the **single-model** serve path (`lobes serve`), but it does **not** protect the
fleet gateway's proxied endpoints (generate, embed, rerank, or `/v1/audio/*`). Keep
the gateway port off the public internet; when exposing it with `lobes tunnel`,
layer Cloudflare Access or an IP allowlist on top. Per-endpoint gateway auth is
planned for a later release.

### Supported catalog vs. warm backends

Two questions that look alike but aren't:

- **What's loaded right now?** — the model(s) actually in GPU memory. The live
  source is `GET /v1/models` (OpenAI-standard; one model in single-model mode; the
  generate primary plus the embedding + reranker gears in the fleet); `lobes fleet
  status` queries it. It changes when you
  `lobes switch` or bring the fleet up/down. (`lobes status` / `lobes whoami`
  instead report the model the deployment is *configured* to serve — from `.env` —
  plus container health, which is configuration, not a live `/v1/models` query.)
- **What's *supported* (what can I warm up)?** — the curated catalog of "gears"
  lobes knows how to serve, from `lobes overview --list` or
  `GET /v1/models/supported`. Each entry is tagged `load-tested` (proven on this
  box) or `configured` (declared, not yet proven). It's **static** — defined in
  `lobes/catalog.py`, shipped in the wheel, unchanged by what's running. On
  the gateway endpoint each entry also carries a runtime-computed `loaded` /
  `default` flag.

Mnemonic: the catalog is *what's on the menu (and which dishes we've cooked)*;
`/v1/models` is *what's hot now*.

## Verbs

```bash
lobes init --fleet --apply        # scaffold compose + .env + Dockerfile.gateway
lobes fleet up --apply            # docker compose up -d --build, wait for gateway /health
lobes fleet status                # each container's state + gateway /health + /v1/models
lobes overview --live             # live dashboard: online / offered / busy + usage + endpoints
lobes fleet down --apply          # docker compose down
```

`lobes fleet up` / `down` are **dry-run by default**; pass `--apply` to commit.
`--compose-dir` overrides the deployment dir (default `$LOBES_DIR` or
`$HOME/.lobes`). `lobes fleet status` is read-only — it reports the *warm*
backend(s) (`/v1/models`); for the full set you can switch to, use
`lobes overview --list` / `/v1/models/supported` (see above).

`lobes overview --live` is the read-only **live dashboard**: it reads the gateway
`/status` (or, against a bare single-model server, that server's `/metrics` +
`/health`) and prints what is **online** (per-backend health), **offered**
(models, task families, endpoints), **busy** (in-flight / queued requests), and
cumulative **usage** (prompt/generation tokens, finished requests by reason). HTTP-only, so it
works against a local deployment or a `lobes tunnel` hostname alike; it degrades
gracefully when a backend or its metrics is unreachable.

**`lobes switch` does not drive the fleet** — it rewrites the single-model
`VLLM_*` keys. Change the fleet primary by editing the fleet `.env`
(`PRIMARY_MODEL` and its `PRIMARY_SERVED_NAME` / `PRIMARY_GPU_MEM_UTIL`
/ `PRIMARY_TOOL_CALL_PARSER` / `PRIMARY_QUANTIZATION`) and re-running `lobes fleet
up --apply`. (A fallback, when wired up, uses the parallel `FALLBACK_*` keys.)

## Memory

The fleet runs **one generate backend by default**: the primary owns the box at
`PRIMARY_GPU_MEM_UTIL=0.6` (~75 GiB of the 128 GB), serving the full 256K context
— the load-tested solo footprint (see findings below). The co-resident embedding
and reranker gears are ~0.6B each at `*_GPU_MEM_UTIL=0.06` (a couple GiB apiece),
so they tuck into the remaining headroom without crowding the primary; what does
**not** co-fit is a second ~30B *generate* model (below). That still leaves room
for the OS and other processes.

`--gpu-memory-utilization` is a fraction of *total* unified memory, computed
independently per vLLM process (they don't coordinate). So **if you add a warm
fallback**, `PRIMARY_GPU_MEM_UTIL` + `FALLBACK_GPU_MEM_UTIL` must sum well under
1.0 — two ~30B NVFP4 models do **not** co-fit a GB10 that is also running other
services (the prior `0.40` + `0.35` co-residence default OOM-looped; that's why
the fallback was removed). **Validate live** (watch `spark memory` / `nvidia-smi`
at `lobes fleet up`; OOM is the top operational risk).

Note the throughput trade-off if you do co-resident two backends: decode is
memory-bandwidth bound and the bandwidth (~273 GB/s) is **shared** — two backends
decoding *simultaneously* split it. The gateway routes one request to one backend,
so a single client sees full speed.

## Live validation findings — DGX Spark (GB10), 2026-05-30

First live `lobes fleet up` of the 27B-primary + 35B-A3B-fallback pair on
`spark-f8a9` (a **shared** box: tritonserver/realtime-api, nova, reachy, mongo
also running, ~12–20 GiB baseline). Measured with `dgx-spark-cli` (`spark`):

| What | Result |
|---|---|
| **27B (primary) solo load → `/health`** | **~423 s (~7 min)**: weight load 160 s (28.25 GiB), profiling/warmup 55 s, CUDA-graph capture + KV ~200 s |
| 27B decode (batch=1, 512 tok) | **8.0 tok/s**; prefill 2,015 tok in 3.29 s |
| 27B footprint | **~75.5 GiB at util 0.6** (≈ 28 GiB weights + 42 GiB KV + 3.7 GiB CUDA graphs) |
| **35B-A3B (old fallback) load** | **Did not complete.** Co-resident: `CUDA error: out of memory` on engine init → 14+ restart crash-loop. Even *solo* (65 GiB free): crashed/stalled at "Loading safetensors 0%", never `/health` in 8+ min. No benchmark obtained. |
| Co-residence (27B + 35B-A3B) | **Not viable on this box.** 27B alone (~75 GiB) + 35B-A3B (~24 GiB weights + KV) + baseline services exceed the 121.7 GiB unified pool → OOM + swap thrash (swap hit 68 %). |
| **Mistral-24B (new fallback) solo load → `/health`** | **Loaded cleanly** (port 8001, util 0.4): 15.05 GiB weights, 30.69 GiB KV, ~49.6 GiB total. Decode **14.9 tok/s**; prefill 2,009 tok in 1.49 s; tool calling ✅. See [`docs/mistral-small-3.2-24b-nvfp4.md`](mistral-small-3.2-24b-nvfp4.md). |

**Conclusion — the "two always-warm *generate* models" premise needs a dedicated
box, so the default is one generate backend.** On a GB10 shared with other
services, two ~30B NVFP4 models do not co-fit with usable KV caches. The default
fleet therefore serves the **Qwen generate primary** at its load-tested solo
headroom (util 0.6, full 256K, ~75 GiB), with the tiny embedding + reranker gears
co-resident (util 0.06 each). If you genuinely need two warm models, run on a dedicated machine,
pair two small models, or wire the opt-in fallback (see "Adding a fallback") and
drop both utils. Single-model `lobes switch` (one warm at a time) remains the
other path.

**Fallback history.** The original 35B-A3B MoE fallback never loaded
([`docs/qwen3.6-35b-a3b-nvfp4.md`](qwen3.6-35b-a3b-nvfp4.md)); it was replaced
(2026-05-30) by the dense `RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4`
(loads reliably, ~15 GiB weights —
[`docs/mistral-small-3.2-24b-nvfp4.md`](mistral-small-3.2-24b-nvfp4.md)). Even the
dense 24B stayed tight on a shared box, so the warm fallback was **removed from
the default fleet** — Mistral remains a selectable catalog candidate and the
documented opt-in fallback. The `0.55`/`0.30` → `0.40`/`0.35` util history above is
the record of that co-residence struggle.

## Coherence with the single-model verbs

The fleet `.env` mirrors `VLLM_MODEL` / `VLLM_SERVED_NAME` / `VLLM_TOOL_CALL_PARSER`
(= the primary's) so the read-only single-model verbs (`lobes status`,
`lobes whoami`, `lobes doctor`'s `env_coherence` check) stay sensible on a fleet
deployment. `culture.yaml`'s `model: vllm-local/sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`
resolves through the gateway on `:8000` as the default.
