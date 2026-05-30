# Fleet: two models behind one OpenAI gateway

The **fleet** runs two always-warm vLLM models behind a single stdlib
OpenAI-compatible gateway, managed by model-gear as three Docker containers. It is
an alternative to the single-model deployment — scaffold it with
`model init --fleet` (the single-model `model init` is unchanged and remains the
default).

## Why

The single-model deployment serves one model on `:8000` and `model switch` swaps
it (freeing the prior model). The fleet instead keeps **both** models loaded and
puts one OpenAI endpoint in front of them, so:

- existing clients (the acp `vllm-local` provider, `curl`, …) keep pointing at
  `:8000` and keep working — an unknown/missing `model` defaults to the primary;
- a second model is addressable by name in the same `/v1/...` calls;
- if the chosen backend is down, the gateway fails over to the other one.

On the DGX Spark (GB10, 128 GB unified memory) the fleet pairs a hybrid-Mamba
**27B** primary with a dense **24B** fallback
(`RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4`) that loads reliably and
decodes a little faster (~15 vs ~10 tok/s). It replaced the Qwen3.6-35B-A3B MoE,
which never reached `/health` on this box (see "Live validation findings").

## Topology

```text
client / acp ──:8000──▶ model-gear-gateway   (python -m model_gear.gateway)
                          │  route by `model` → default → failover
                          ├──▶ model-gear-vllm-primary   :8000 (internal)
                          └──▶ model-gear-vllm-fallback  :8000 (internal)
```

Three containers, all `restart: unless-stopped`:

| Container | Role | Host port |
|---|---|---|
| `model-gear-gateway` | stdlib reverse proxy (the single OpenAI front) | `${VLLM_PORT:-8000}` |
| `model-gear-vllm-primary` | primary model (default: `mmangkad/Qwen3.6-27B-NVFP4`) | internal only |
| `model-gear-vllm-fallback` | dense fallback (default: `RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4`) | internal only |

The backends are reachable only on the compose network
(`http://vllm-primary:8000`, `http://vllm-fallback:8000`); only the gateway is
published to the host. The gateway needs no Docker socket access — compose owns
the lifecycle; the gateway only routes.

## The gateway

A pure-stdlib (`http.server` + `http.client`, no third-party deps) reverse proxy:

- **Name routing** — a request's `model` routes to the backend that serves it,
  plus any `GATEWAY_ALIASES`. The forwarded body's `model` is rewritten to the
  backend's `--served-model-name` so the backend accepts aliased/default routes.
- **Default model** — a missing or unknown `model` routes to
  `GATEWAY_DEFAULT_MODEL` (the primary).
- **Failover** — if the chosen backend refuses the connection or returns a 5xx
  **before any response body**, the request is retried against the other backend.
  A 4xx is a client error (returned verbatim, no failover). Once a 2xx body starts
  streaming there is no retry — the client already has bytes.
- **Streaming** — `"stream": true` (SSE) is relayed chunk-by-chunk with per-chunk
  flushing; normal JSON is buffered with `Content-Length`.
- **Endpoints** — `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`
  (proxied), `/v1/models` (OpenAI-standard, lists the two loaded backends),
  `/v1/models/supported` (the full supported-model catalog — every gear you can
  change to, each flagged `loaded` / `default`), `/health` (gateway liveness).
  See [Supported catalog vs. warm backends](#supported-catalog-vs-warm-backends)
  for what `/v1/models` and `/v1/models/supported` each mean.

The gateway image is built from the scaffolded `Dockerfile.gateway`
(`pip install model-gear==${MODEL_GEAR_VERSION}`, as a non-root user); `model init
--fleet` pins `MODEL_GEAR_VERSION` to the running model-gear release. The version
is required (pinning keeps the image reproducible); from-source/dev boxes that run
ahead of a PyPI release point `MODEL_GEAR_VERSION` at a published TestPyPI `.devN`
build.

### Supported catalog vs. warm backends

Two questions that look alike but aren't:

- **What's loaded right now?** — the model(s) actually in GPU memory. The live
  source is `GET /v1/models` (OpenAI-standard; one model in single-model mode, both
  backends in the fleet); `model fleet status` queries it. It changes when you
  `model switch` or bring the fleet up/down. (`model status` / `model whoami`
  instead report the model the deployment is *configured* to serve — from `.env` —
  plus container health, which is configuration, not a live `/v1/models` query.)
- **What's *supported* (what can I warm up)?** — the curated catalog of "gears"
  model-gear knows how to serve, from `model overview --list` or
  `GET /v1/models/supported`. Each entry is tagged `load-tested` (proven on this
  box) or `configured` (declared, not yet proven). It's **static** — defined in
  `model_gear/catalog.py`, shipped in the wheel, unchanged by what's running. On
  the gateway endpoint each entry also carries a runtime-computed `loaded` /
  `default` flag.

Mnemonic: the catalog is *what's on the menu (and which dishes we've cooked)*;
`/v1/models` is *what's hot now*.

## Verbs

```bash
model init --fleet --apply        # scaffold compose + .env + Dockerfile.gateway
model fleet up --apply            # docker compose up -d --build, wait for gateway /health
model fleet status                # each container's state + gateway /health + /v1/models
model fleet down --apply          # docker compose down
```

`model fleet up` / `down` are **dry-run by default**; pass `--apply` to commit.
`--compose-dir` overrides the deployment dir (default `$MODEL_GEAR_DIR` or
`~/.model-gear`). `model fleet status` is read-only — it reports the *warm*
backends (`/v1/models`); for the full set you can switch to, use
`model overview --list` / `/v1/models/supported` (see above).

**`model switch` does not drive the fleet** — it rewrites the single-model
`VLLM_*` keys. Change fleet models by editing the fleet `.env`
(`PRIMARY_MODEL` / `FALLBACK_MODEL` and their `*_SERVED_NAME` / `*_GPU_MEM_UTIL`
/ `*_TOOL_CALL_PARSER` / `*_QUANTIZATION`) and re-running `model fleet up --apply`.

## Memory (both warm)

Both models stay resident, so `PRIMARY_GPU_MEM_UTIL` + `FALLBACK_GPU_MEM_UTIL`
must sum well under 1.0 of the 128 GB. The scaffolded defaults are **0.40** +
**0.35** (≈ 96 GB reserved on a dedicated box, leaving ~32 GB for the OS, other
processes, and the load/warmup spike). These are **estimates that require a
dedicated box** — `--gpu-memory-utilization` is a fraction of *total* unified
memory and each vLLM process computes it independently (they don't coordinate),
so on a GB10 that is also running other services the two backends OOM. **Validate
live** (watch `spark memory` / `nvidia-smi` at `model fleet up`; OOM is the top
operational risk) and lower further on a shared box. See the findings below.

Note the throughput trade-off: decode is memory-bandwidth bound and the bandwidth
(~273 GB/s) is **shared**. The MoE reads only its active experts per token, so it
stays fast; two backends decoding *simultaneously* split the bandwidth. The
gateway routes one request to one backend, so a single client sees full speed.

## Live validation findings — DGX Spark (GB10), 2026-05-30

First live `model fleet up` of the 27B-primary + 35B-A3B-fallback pair on
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

**Conclusion — the "two always-warm models" premise needs a dedicated box.** On a
GB10 shared with other services, two ~30B NVFP4 models do not co-fit with usable
KV caches. Options: (a) run the fleet on a dedicated machine; (b) pair two
genuinely small models; or (c) use single-model `model switch` (one warm at a
time) instead of the fleet. The `0.55`/`0.30` default shipped in 0.10.0 OOM-looped
the fallback and was corrected to `0.40`/`0.35` (a dedicated-box estimate). The
35B-A3B's own load instability (crash/stall even solo) is tracked in
[`docs/qwen3.6-35b-a3b-nvfp4.md`](qwen3.6-35b-a3b-nvfp4.md).

**Fallback default changed to a dense 24B (2026-05-30).** Because the 35B-A3B
never loaded, the default fallback is now
`RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4` — dense, loads reliably, and
smaller (~15 GiB weights vs the 27B's ~28 GiB), which also makes co-residence less
of a stretch. Even so, two warm models on a *shared* GB10 remains tight; the
dedicated-box guidance above still stands.

## Coherence with the single-model verbs

The fleet `.env` mirrors `VLLM_MODEL` / `VLLM_SERVED_NAME` / `VLLM_TOOL_CALL_PARSER`
(= the primary's) so the read-only single-model verbs (`model status`,
`model whoami`, `model doctor`'s `env_coherence` check) stay sensible on a fleet
deployment. `culture.yaml`'s `model: vllm-local/mmangkad/Qwen3.6-27B-NVFP4`
resolves through the gateway on `:8000` as the default.
