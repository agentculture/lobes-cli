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

On the DGX Spark (GB10, 128 GB unified memory) both ~30B-class NVFP4 models fit at
once; the fleet pairs a hybrid-Mamba **27B** primary with an **MoE** fallback
(`A3B` ≈ 3B active params) that decodes much faster, so the fast model stays fast.

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
| `model-gear-vllm-fallback` | MoE fallback (default: `mmangkad/Qwen3.6-35B-A3B-NVFP4`) | internal only |

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

The gateway image is built from the scaffolded `Dockerfile.gateway`
(`pip install model-gear==${MODEL_GEAR_VERSION}`, as a non-root user); `model init
--fleet` pins `MODEL_GEAR_VERSION` to the running model-gear release. The version
is required (pinning keeps the image reproducible); from-source/dev boxes that run
ahead of a PyPI release point `MODEL_GEAR_VERSION` at a published TestPyPI `.devN`
build.

## Verbs

```bash
model init --fleet --apply        # scaffold compose + .env + Dockerfile.gateway
model fleet up --apply            # docker compose up -d --build, wait for gateway /health
model fleet status                # each container's state + gateway /health + /v1/models
model fleet down --apply          # docker compose down
```

`model fleet up` / `down` are **dry-run by default**; pass `--apply` to commit.
`--compose-dir` overrides the deployment dir (default `$MODEL_GEAR_DIR` or
`~/.model-gear`). `model fleet status` is read-only.

**`model switch` does not drive the fleet** — it rewrites the single-model
`VLLM_*` keys. Change fleet models by editing the fleet `.env`
(`PRIMARY_MODEL` / `FALLBACK_MODEL` and their `*_SERVED_NAME` / `*_GPU_MEM_UTIL`
/ `*_TOOL_CALL_PARSER` / `*_QUANTIZATION`) and re-running `model fleet up --apply`.

## Memory (both warm)

Both models stay resident, so `PRIMARY_GPU_MEM_UTIL` + `FALLBACK_GPU_MEM_UTIL`
must sum well under 1.0 of the 128 GB. The scaffolded defaults are **0.55** +
**0.30** (≈ 109 GB reserved, leaving headroom for the OS and KV growth). The 27B
primary is heavier than the old 32B (~70 GB at util 0.6), so its share rises and
the fallback's drops to keep the sum in a safe band. These are estimates for a
hybrid-Mamba 27B + a 35B-A3B MoE — **validate live** (watch `nvidia-smi` at
`model fleet up`; OOM is the top operational risk) and tune the two values.

Note the throughput trade-off: decode is memory-bandwidth bound and the bandwidth
(~273 GB/s) is **shared**. The MoE reads only its active experts per token, so it
stays fast; two backends decoding *simultaneously* split the bandwidth. The
gateway routes one request to one backend, so a single client sees full speed.

## Coherence with the single-model verbs

The fleet `.env` mirrors `VLLM_MODEL` / `VLLM_SERVED_NAME` / `VLLM_TOOL_CALL_PARSER`
(= the primary's) so the read-only single-model verbs (`model status`,
`model whoami`, `model doctor`'s `env_coherence` check) stay sensible on a fleet
deployment. `culture.yaml`'s `model: vllm-local/mmangkad/Qwen3.6-27B-NVFP4`
resolves through the gateway on `:8000` as the default.
