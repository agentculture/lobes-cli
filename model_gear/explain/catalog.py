"""Markdown catalog for ``model explain <path>``.

Each entry is verbatim markdown. Keys are topic-path tuples. The empty tuple and
``("model-gear",)`` both resolve to the root entry (aliased).

Keep bodies self-contained — an agent reading a single entry should get enough
context without chaining reads.
"""

from __future__ import annotations

_ROOT = """\
# model-gear

model-gear is the tooling that **runs, assesses, and switches** the local,
OpenAI-compatible vLLM model the Culture mesh consumes. The binary is `model`.

The served model is what the **model-gear** agent connects to over the acp
`vllm-local` provider — the same model-gear runs the engine and consumes it (the
tool and the deployed agent share one identity).

## Verbs

- `model init [TARGET]` — scaffold a deployment dir (default `~/.model-gear`).
  Dry-run by default; `--apply` writes, `--force` overwrites.
- `model serve` (alias `start`) / `model stop` — start / stop the vLLM server.
  Dry-run by default; `--apply` to commit.
- `model switch <model>` — switch the served model. Dry-run by default;
  `--apply` recreates the container and waits for `/health`.
- `model fleet up|down|status` — drive the gateway fleet (one OpenAI front over
  the generate primary plus co-resident embedding + reranker gears, routed by task
  family; a generate fallback is opt-in). Scaffold it with `model init --fleet`.
  `up`/`down` are dry-run by default; `--apply` to commit.
- `model tunnel` — expose the local API at a public hostname via a Cloudflare
  Tunnel (`--stop` to tear down). Dry-run by default; `--apply` to commit.
- `model status` — read-only: the configured served model (from `.env`), container
  state, `/health`. (For the full set you can switch to, use `model overview --list`;
  for what's actually loaded now, the live `/v1/models`.)
- `model assess` — read-only correctness probes + reasoning-trace detection.
- `model benchmark` — read-only decode throughput + prefill latency.
- `model overview` — snapshot of the tool, the served model, and the supported
  catalog (the gears you can switch to). `--current` = configured served model;
  `--list` = catalog.
- `model whoami` — tool, machine, served model, container health.
- `model doctor` — diagnose docker / compose / `.env` / health.

## Mutation safety

Write verbs (`switch`, `serve`, `stop`, `init`) are **dry-run by default** and
require `--apply` to commit. The rest are read-only.

## Exit-code policy

- `0` success
- `1` user-input error (bad flag, bad path, missing arg)
- `2` environment / setup error (docker missing, `.env` unreadable, endpoint down)
- `3+` reserved

## See also

- `model explain switch`
- `model explain tuning` (purpose + machine profiles)
- `model explain fleet`
- `model explain gateway`
- `model explain tunnel` (expose the API from anywhere)
- `model explain assess`
- `model explain backend`
- `model explain models`
- `model explain embeddings` (POST /v1/embeddings — 1024-dim Qwen3 embedder)
- `model explain rerank` (POST /v1/rerank — Jina/Cohere reranking)
- `model explain score` (POST /v1/score — cross-encoder raw scoring)
"""

_SWITCH = """\
# model switch

`model switch <model>` changes which vLLM model is served. **Dry-run by
default** (prints the plan, changes nothing); `--apply` commits.

On `--apply` it resolves a serve config from three layers — the **machine**
profile (`--machine`, default auto-detected), the **workload** profile
(`--purpose`, default `balanced`), and the model's catalog entry (quantization,
tool parser) — and writes the `VLLM_*` vars to the deployment `.env` —

```text
VLLM_MODEL, VLLM_SERVED_NAME, VLLM_PORT, VLLM_PURPOSE, VLLM_MACHINE,
VLLM_MAX_MODEL_LEN, VLLM_GPU_MEM_UTIL, VLLM_ATTENTION_BACKEND,
VLLM_MAX_NUM_SEQS, VLLM_MAX_NUM_BATCHED_TOKENS
(+ VLLM_TOOL_CALL_PARSER / VLLM_QUANTIZATION for a known/overridden model)
```

— then recreates the container (`docker compose down && up -d`) and waits for
`/health` (the first run downloads weights, so this can take many minutes).

Flags: `--port`, `--purpose {balanced,prompt-heavy,decode-heavy}` (tunes batching
+ the benchmark shape), `--machine {auto,spark,thor,blackwell,generic}` (GPU mem /
context / attention defaults), `--max-model-len` / `--gpu-mem-util` (explicit
overrides of the machine profile), `--served-name`, `--tool-call-parser` (e.g.
`hermes` for Qwen3 dense, `qwen3_coder` for Qwen3-Coder/3.6), `--quantization`,
`--compose-dir`, `--apply`, `--json`. Switching to a model with catalog serve-extras
prints a reminder for the compose-only flags that can't be defaulted in the shared
template: the Qwen3.6-35B-A3B MoE wants `--moe-backend=marlin` (its MTP
speculative-config is *not* carried — it fails to load on that checkpoint), and the
`sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` candidate carries an MTP
`--speculative-config` (plus `--trust-remote-code` / `--language-model-only`).
Only one ~30B-class model fits on a single GB10 at a time, so the switch frees
the prior model before starting the new one.
"""

_SERVE = """\
# model serve / model stop

`model serve` (alias `start`) runs `docker compose up -d` in the deployment dir
and waits for `/health`. `model stop` runs `docker compose down`. Both are
**dry-run by default**; pass `--apply` to commit. `--compose-dir` overrides the
deployment dir (default `$MODEL_GEAR_DIR` or `~/.model-gear`).
"""

_STATUS = """\
# model status

Read-only snapshot of the current deployment: the configured `VLLM_MODEL` /
`VLLM_SERVED_NAME` / `VLLM_PORT` (from `.env`), the `model-gear-vllm` container's
lifecycle + health state, and whether `/health` is responding. Supports
`--json`. This reports the *configured* served model (from `.env`) + health — not
a live `/v1/models` query, so for what's actually loaded now query `/v1/models`,
and for the full set you can switch to use `model overview --list`.
"""

_ASSESS = """\
# model assess

Read-only **correctness** probes against the served model, emitted as a markdown
block ready to paste into a per-model doc under `docs/`:

- `17 * 23 = 391`
- a train leaving 14:45 arriving 17:10 takes `145` minutes

It also detects which field carried the reasoning trace (`reasoning` on the
nv26.04 vLLM build, `reasoning_content` on older builds) and reports its length,
plus host-side facts (image tag, GPU memory). Throughput lives in
`model benchmark`. Supports `--json`.

`--tools` adds an OpenAI tool-calling probe: a `tool_choice:"auto"` request must
return a `tool_calls` array naming a `finish` function (degrades gracefully to a
FAIL row if the server lacks `--enable-auto-tool-choice`).
"""

_BENCHMARK = """\
# model benchmark

Read-only **throughput** measurement, emitted as a markdown block for a per-model
doc. The workload shape is the active **purpose** — it defaults to the configured
`VLLM_PURPOSE` (so the numbers track the serve config) and is overridable with
`--purpose {balanced,prompt-heavy,decode-heavy}` or explicit `--input-len` /
`--output-len`. Measures decode throughput (the output length forced over
`--runs` repetitions, batch=1 greedy) and prefill latency (a prompt sized to the
input length). Reports host-side facts (image tag, GPU memory) too. Correctness
lives in `model assess`. Supports `--json`.
"""

_INIT = """\
# model init

`model init [TARGET]` scaffolds a deployment directory by copying the packaged
`docker-compose.yml` and `env.example`→`.env`. `TARGET` defaults to
`~/.model-gear`; pass a path, or `.` for the current folder. **Dry-run by
default** (lists what it would write); `--apply` writes, `--force` overwrites
existing files. Supports `--json`.
"""

_BACKEND = """\
# model-gear backend

model-gear runs a **local vLLM server** that exposes an OpenAI-compatible API.
The Culture `acp` backend (opencode's `vllm-local` provider) connects to it —
this is the model the **model-gear** agent consumes.

## Deployment

`docker-compose.yml` runs the NGC vLLM image (`nvcr.io/nvidia/vllm:26.04-py3`)
as the `model-gear-vllm` container. `model init` scaffolds it into
`~/.model-gear`; `model serve` brings it up. Key serve flags:

```text
--quantization=modelopt_fp4   # nvidia/ checkpoints are ModelOpt FP4
--kv-cache-dtype=fp8
--reasoning-parser=qwen3      # expose the <think> trace
--enable-auto-tool-choice     # OpenAI tool/function calling (tool_choice:"auto")
--tool-call-parser=hermes     # VLLM_TOOL_CALL_PARSER; qwen3_coder for Qwen3-Coder/3.6
--enable-prefix-caching
```

Tuned for DGX Spark (GB10 Grace Blackwell, 128 GB unified memory) via
`VLLM_GPU_MEM_UTIL` (default 0.6) and `VLLM_MAX_MODEL_LEN` (default 262144).

## The must-match invariant

`VLLM_SERVED_NAME` in `.env` **must equal** the part after `vllm-local/` in
the `culture.yaml` `model:` field, or the acp provider won't resolve the
model. `model doctor` checks this.
"""

_MODELS = """\
# model-gear models

Per-model notes live under `docs/` — one markdown file per model that has been
run on this hardware, holding the correctness + throughput numbers produced by
`model assess` and `model benchmark`.

- `docs/qwen3.6-27b-text-nvfp4-mtp.md` — `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`,
  the fleet's **default primary** (promoted 2026-05-31): the 27B re-exported with its
  MTP draft head restored so vLLM speculative decoding works (the baseline NVFP4
  export drops it). Text-only; its MTP serve flags are baked into the compose
  template. Load-tested on the GB10: ~18.7-19.1 tok/s decode (~2.4x the archived
  baseline 27B) at ~72-79% MTP acceptance, tool calling + reasoning verified (#26).
- `docs/qwen3.6-27b-nvfp4.md` — `mmangkad/Qwen3.6-27B-NVFP4`, the **archived** former
  primary (hybrid Mamba/linear-attn + ViT, 256K native). Retained as the MTP
  primary's tokenizer source and the only vision-capable 27B.
- `docs/mistral-small-3.2-24b-nvfp4.md` — `RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4`,
  the dense fallback the gateway fleet pairs with the primary (loads reliably on
  the GB10; serve with the mistral tokenizer + images disabled — required for
  tool-call parsing on this build; see the doc).
- `docs/qwen3-32b-nvfp4.md` — `nvidia/Qwen3-32B-NVFP4`, a dense candidate (faster
  decode; swap in via `PRIMARY_MODEL` / `model switch`).
- `docs/qwen3.6-35b-a3b-nvfp4.md` — `mmangkad/Qwen3.6-35B-A3B-NVFP4`, a MoE
  candidate (the former fallback; OOM'd/stalled on the GB10, never load-tested).
- `docs/qwen3-embedding-0.6b.md` — `Qwen/Qwen3-Embedding-0.6B`, the **embedding
  gear**: 0.6B dense text embedder, 1024-dim (Matryoshka-truncatable), 32K context,
  served via `/v1/embeddings` (`--runner pooling --convert embed`). Warm fleet
  backend — co-resident with the 27B primary on the GB10.
- `docs/qwen3-reranker-0.6b.md` — `Qwen/Qwen3-Reranker-0.6B`, the **reranker
  gear**: 0.6B cross-encoder, served via `/v1/rerank` + `/v1/score`
  (`--runner pooling --convert classify`). Same backend handles both endpoints.
  Warm fleet backend — co-resident on the GB10.

These models *are* the **supported catalog** — the gears you can switch to, each
tagged `load-tested` (proven on this box) or `configured` (declared, not yet
proven). It is static (defined in `model_gear/catalog.py`, shipped in the wheel).
Read it with `model overview --list` or the gateway's `GET /v1/models/supported`;
it flags which one is currently served.

For what is *loaded right now* (in GPU memory this instant) use the live
`/v1/models` (which `model fleet status` queries) — that is runtime truth, not the
catalog. (`model status` / `model whoami` report the *configured* served model from
`.env` + health, not a live list.) Mnemonic: the catalog is what's on the menu;
`/v1/models` is what's hot now. See `model explain gateway` for the endpoint split,
and `model explain fleet` to run two side-by-side behind one OpenAI endpoint.
"""

_FLEET = """\
# model fleet

The fleet runs the **always-warm Qwen primary plus co-resident embedding and
reranker gears behind one OpenAI-compatible gateway**, managed as four containers
by default: `model-gear-vllm-primary`, `model-gear-vllm-embed`,
`model-gear-vllm-rerank`, and `model-gear-gateway` (a warm *generate* fallback,
`model-gear-vllm-fallback`, is opt-in). The gateway routes each request to the
right backend by task family (generate / embed / score / rerank). Scaffold it
with `model init --fleet` (writes the fleet `docker-compose.yml`, `.env`, and
`Dockerfile.gateway`), then:

- `model fleet up` — `docker compose up -d --build` (builds the gateway image),
  then waits for the gateway `/health`. The vLLM backend loads in the background.
- `model fleet down` — `docker compose down`.
- `model fleet status` — read-only: each container's state, the gateway `/health`,
  and the routed model list (`/v1/models`).

`up`/`down` are **dry-run by default**; pass `--apply` to commit. `--compose-dir`
overrides the deployment dir. There is **one generate backend** by default, so the
primary runs at its solo headroom (`PRIMARY_GPU_MEM_UTIL=0.6`, full 256K); the
embedding + reranker gears are ~0.6B (`*_GPU_MEM_UTIL=0.06` each), so they
co-reside without crowding it. If you add a warm *generate* fallback, set
`PRIMARY_GPU_MEM_UTIL` + `FALLBACK_GPU_MEM_UTIL` to sum well under 1.0 (they share
the 128 GB unified memory).

Note: `model switch` does **not** drive the fleet (it rewrites the single-model
`VLLM_*` keys). Change the fleet primary by editing the fleet `.env` and
re-running `model fleet up --apply`. See `model explain gateway` for routing.
"""

_GATEWAY = """\
# model-gear gateway

The gateway is a stdlib (no third-party deps) OpenAI-compatible reverse proxy
that fronts the fleet's vLLM backend(s) on one port — the host port the acp
`vllm-local` provider already expects. It runs as the `model-gear-gateway`
container (`python -m model_gear.gateway`).

## Routing

- **By name** — a request's `model` field routes to the backend that serves it
  (plus any `GATEWAY_ALIASES`). The forwarded body's `model` is rewritten to the
  backend's `--served-model-name` so the backend accepts it.
- **Default** — a missing or unknown `model` routes to `GATEWAY_DEFAULT_MODEL`
  (the primary), so existing single-model clients keep working unchanged.
- **Failover** — if the chosen backend refuses the connection or returns a 5xx
  **before any response body**, the gateway retries the request against the other
  backend. A 4xx (client error) is returned verbatim — no failover. Once a 2xx
  body starts streaming, there is no retry. SSE streams (`"stream": true`) are
  relayed chunk-by-chunk with per-chunk flushing.

## Endpoints

- `/v1/chat/completions`, `/v1/completions` — chat and completion requests;
  routed to the primary (or fallback, if configured) by `model` field.
- `/v1/embeddings` — dense text embeddings; served by the warm
  **Qwen3-Embedding-0.6B** fleet backend, routed by `model` field
  (`"model": "Qwen/Qwen3-Embedding-0.6B"`).
- `/v1/rerank` — Jina/Cohere-compatible re-ranking; served by the warm
  **Qwen3-Reranker-0.6B** fleet backend, routed by `model` field.
- `/v1/score` — vLLM cross-encoder raw scoring; same warm **Qwen3-Reranker-0.6B**
  backend as `/v1/rerank`, routed by `model` field.
- `/v1/models` — OpenAI-standard model list (lists all loaded backends).
- `/v1/models/supported` — the full supported-model catalog (every gear you can
  switch to, each flagged `loaded` / `default`).
- `/health` — gateway liveness check.

All endpoints are reached at the same gateway port — routing is by the request's
`model` field. Configured via the `gateway` service's environment in the fleet
compose (`PRIMARY_URL` / `FALLBACK_URL` / `*_SERVED_NAME` / `GATEWAY_DEFAULT_MODEL`
/ `GATEWAY_ALIASES` / timeouts).
"""

_TUNNEL = """\
# model tunnel

`model tunnel` exposes the local OpenAI-compatible vLLM API (`127.0.0.1:8000`) at
an owner-chosen public hostname through a **Cloudflare Tunnel**, so Culture agents
can call it from anywhere as an ordinary provider (`base_url` + `api_key`).
**Dry-run by default** (prints the exact `cloudflared` command — plaintext tokens
redacted — and the public `https://<host>/v1` URL); `--apply` starts a standalone
`cloudflared tunnel run` in the background (logging to `cloudflared.log` in the
deployment dir), and `--stop --apply` terminates it.

## Config (never committed)

- **Hostname** — `--hostname`, else `$CULTURE_VLLM_PUBLIC_HOSTNAME`, else
  `CULTURE_VLLM_PUBLIC_HOSTNAME` in the gitignored `.cf-tunnel.env` (deployment dir).
- **Run-token** — from `.cf-tunnel.env`: `CULTURE_CF_TUNNEL_TOKEN_SHUSHU` (a
  shushu-sealed secret name, preferred) or `CULTURE_CF_TUNNEL_TOKEN` (plaintext
  fallback). `model init` scaffolds `cf-tunnel.env.example`; copy it to
  `.cf-tunnel.env` and edit.

## Two-step flow

1. **Cloudflare side, once** — the sibling `cultureflare` tool provisions the
   tunnel + ingress + DNS and seals the run-token:
   `cultureflare remote-login setup --hostname <host> --service http://127.0.0.1:8000
   --no-access --shushu --apply`.
2. **Local side** — `model serve --apply` (with `CULTURE_VLLM_API_KEY` set in
   `.env` so the API is bearer-gated) then `model tunnel --apply`.

`--apply` preflights that `cloudflared` (and `shushu`, for the sealed token) is on
PATH and that the local server answers `/health` first. **Set `CULTURE_VLLM_API_KEY`
before exposing the API** — without it the tunnel publishes an unauthenticated model.
See `model explain backend` and the README "Expose the API" section.
"""

_WHOAMI = """\
# model whoami

The smallest identity probe. Reports model-gear's view: the `tool` + `version`,
the `machine` (hostname + GPU), the currently-`served_model` and `port` (read
from the deployment `.env`), the `container_health`, and the `agent` that
consumes the model (`model-gear`, from `culture.yaml`). Read-only; supports
`--json`. `served_model` is the *configured* served model (from `.env`), not a live
`/v1/models` query — see `model overview --list` for the full supported catalog you
can switch to.
"""

_LEARN = """\
# model learn

Prints a structured self-teaching prompt: purpose, the command map, the mutation
-safety rule, the `--json` contract, and the exit-code policy. Enough shape for
an agent to author its own usage skill without scraping `--help`. Supports
`--json`.
"""

_EXPLAIN = """\
# model explain

Resolves a topic path against the markdown catalog and prints the body. With no
path it returns the root overview (same as `model explain model-gear`). Unknown
paths exit `1` with a `hint:` pointing back at the root. Supports `--json`, which
wraps the markdown as `{"path": [...], "markdown": "..."}`.
"""

_OVERVIEW = """\
# model overview

A read-only snapshot of model-gear: identity (tool / version / machine), the verb
surface, capabilities, the configured served model (from `.env`), and the
**supported catalog** — the gears you can switch to (`model_gear/catalog.py`, each tagged
`load-tested` / `configured`). `--current` shows only the configured served-model
block (from `.env`); `--list` shows only the catalog (the same set as the gateway's
`/v1/models/supported`; for what's actually *loaded* now, query the live
`/v1/models`).
`model cli overview` is the parallel snapshot of the CLI surface itself. Supports
`--json` (`{"subject", "sections"}`). A stray path argument is accepted and
ignored, so `overview <path>` never hard-fails.
"""

_DOCTOR = """\
# model doctor

Diagnoses the deployment with real checks: `docker_available` (docker + compose
resolve), `compose_present` (a deployment is scaffolded), `env_coherence`
(`.env` has `VLLM_SERVED_NAME` and it matches `culture.yaml`), and
`health_reachable` (`/health` responds). A down model is a *warning*, not a
failure — only missing docker or an un-scaffolded deployment make the run exit
non-zero. JSON contract: `{"healthy", "checks"}`. Supports `--json`.
"""

_EMBEDDINGS = """\
# model explain embeddings

`POST /v1/embeddings` — OpenAI-compatible text embeddings served by the warm
**Qwen3-Embedding-0.6B** fleet backend, routed by model name through the gateway.

## Request

```json
{
  "model": "Qwen/Qwen3-Embedding-0.6B",
  "input": ["text a", "text b"]
}
```

`input` accepts a string or a list of strings. The served name `Qwen/Qwen3-Embedding-0.6B`
is the catalog id — the gateway routes to this backend by matching the `model` field.

Optional: `"dimensions": 512` — truncate the output embedding to any Matryoshka
sub-dimension (32 / 64 / 128 / 256 / 512 / 768 / 1024). Enabled via the
`--hf-overrides '{"is_matryoshka": true, "matryoshka_dimensions": [...]}'` serve flag.
Omit to get the native **1024-dim** output.

## Response

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

## Key facts

- **Dimension:** 1024 (native); request `"dimensions": N` for Matryoshka truncation.
- **Context:** 32K native, served at `--max-model-len 8192` (tiny KV footprint).
- **Serving:** `--runner pooling --convert embed` (vLLM pooling mode, not a chat
  model; this build's replacement for the old `--task embed`).
- **Served name == catalog id:** `Qwen/Qwen3-Embedding-0.6B`.
- **Warm fleet backend:** co-resident with the 27B primary on the GB10; its small
  `--max-model-len` keeps the KV footprint tiny so all three backends (primary +
  embedder + reranker) co-fit in 128 GB unified memory.
- **Gateway port:** same port as chat — the gateway routes by `model` field.
- **No quantization flag, no tool parser** — pooling model, not a chat model.

## See also

- `model explain rerank` — reranking via `/v1/rerank`
- `model explain score` — raw cross-encoder scoring via `/v1/score`
- `model explain models` — full model catalog
"""

_RERANK = """\
# model explain rerank

`POST /v1/rerank` — Jina / Cohere-compatible re-ranking served by the warm
**Qwen3-Reranker-0.6B** fleet backend (same backend as `/v1/score` — vLLM
`--runner pooling --convert classify`), routed by model name through the gateway.

## Request

```json
{
  "model": "Qwen/Qwen3-Reranker-0.6B",
  "query": "What is the capital of France?",
  "documents": ["Paris is the capital.", "Berlin is the capital.", "Rome is the capital."]
}
```

`query` is a string; `documents` is a list of strings. The gateway routes to this
backend by matching `"model": "Qwen/Qwen3-Reranker-0.6B"` — the catalog id and
served name are the same.

## Response

Results are sorted **best-first** (highest relevance score first). The `index`
refers to the position in the original `documents` list.

```json
{
  "results": [
    {"index": 0, "relevance_score": 0.91},
    {"index": 2, "relevance_score": 0.18},
    {"index": 1, "relevance_score": 0.07}
  ]
}
```

## Key facts

- **Shape:** Jina / Cohere `/v1/rerank` — sorted by relevance, best-first.
- **Backend:** `Qwen3-Reranker-0.6B` with `--runner pooling --convert classify`
  (cross-encoder via the `Qwen3ForSequenceClassification` hf-override).
- **Rerank + score share one backend** — `/v1/rerank` and `/v1/score` both route
  to the same running container; `/v1/rerank` applies the Jina/Cohere sort + shape.
- **Context:** 32K native, served at `--max-model-len 8192`.
- **Warm fleet backend:** co-resident on the GB10; tiny KV footprint (0.6B, 32K window).
- **Gateway port:** same port as chat — routed by `model` field.

## See also

- `model explain score` — raw pairwise scoring via `/v1/score`
- `model explain embeddings` — dense embeddings via `/v1/embeddings`
- `model explain gateway` — how routing works
"""

_SCORE = """\
# model explain score

`POST /v1/score` — OpenAI / vLLM cross-encoder scoring served by the warm
**Qwen3-Reranker-0.6B** fleet backend (same backend as `/v1/rerank` — vLLM
`--runner pooling --convert classify`), routed by model name through the gateway.

## Request

```json
{
  "model": "Qwen/Qwen3-Reranker-0.6B",
  "text_1": "What is the capital of France?",
  "text_2": ["Paris is the capital.", "Berlin is the capital."]
}
```

`text_1` is the query string; `text_2` is a string or list of strings (the passages
to score). The gateway routes to the backend by matching the `model` field.

## Response

Results are returned in input order (not sorted — use `/v1/rerank` for sorted output).

```json
{
  "object": "list",
  "data": [
    {"index": 0, "score": 0.91},
    {"index": 1, "score": 0.07}
  ]
}
```

## Key facts

- **Shape:** vLLM `/v1/score` — raw scores in input order (no sorting).
- **Backend:** `Qwen3-Reranker-0.6B` with `--runner pooling --convert classify`
  (cross-encoder via the `Qwen3ForSequenceClassification` hf-override).
- **Rerank + score share one backend** — `/v1/score` and `/v1/rerank` both route
  to the same running container; use `/v1/rerank` for Jina/Cohere sorted output.
- **Context:** 32K native, served at `--max-model-len 8192`.
- **Warm fleet backend:** co-resident on the GB10; tiny KV footprint (0.6B, 32K window).
- **Gateway port:** same port as chat — routed by `model` field.

## See also

- `model explain rerank` — sorted re-ranking via `/v1/rerank`
- `model explain embeddings` — dense embeddings via `/v1/embeddings`
- `model explain gateway` — how routing works
"""

_TUNING = """\
# model tuning — purpose + machine profiles

`model switch` resolves the serve config from three layers (explicit flags win):

1. **machine** (`--machine`, default auto-detected from nvidia-smi + hostname) →
   `VLLM_GPU_MEM_UTIL`, `VLLM_MAX_MODEL_LEN`, `VLLM_ATTENTION_BACKEND`.
   `spark` 0.6/262144 (shared GB10), `blackwell` 0.85/65536 (dedicated VRAM),
   `thor` 0.6/32768, `generic` 0.6/32768. spark is load-tested; the rest are
   configured estimates.
2. **purpose** (`--purpose`, default `balanced`) → `VLLM_MAX_NUM_SEQS`,
   `VLLM_MAX_NUM_BATCHED_TOKENS`, and the shape `model benchmark` exercises:
   `balanced` 4/8192 (≈1K in/1K out), `prompt-heavy` 4/16384 (≈8K in/1K out),
   `decode-heavy` 8/4096 (≈1K in/8K out).
3. **model** (the catalog) → `VLLM_QUANTIZATION`, `VLLM_TOOL_CALL_PARSER`, and a
   printed reminder for any compose-only serve-extras (can't be defaulted in the
   shared template): `--moe-backend=marlin` for the MoE candidate (whose own MTP
   speculative-config is not carried — it fails to load on that checkpoint), and the
   MTP `--speculative-config` for the `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`
   candidate (plus its `--trust-remote-code` / `--language-model-only`).

`model benchmark` defaults its workload shape to the configured `VLLM_PURPOSE`, so
the numbers track the serve config. Override with `--purpose` / `--input-len` /
`--output-len`. The throughput flags follow shahizat's cross-machine NVFP4
benchmark — see `docs/tuning-profiles.md`.
"""

ENTRIES: dict[tuple[str, ...], str] = {
    (): _ROOT,
    ("model-gear",): _ROOT,
    ("model",): _ROOT,
    ("switch",): _SWITCH,
    ("tuning",): _TUNING,
    ("purpose",): _TUNING,
    ("machine",): _TUNING,
    ("serve",): _SERVE,
    ("stop",): _SERVE,
    ("fleet",): _FLEET,
    ("gateway",): _GATEWAY,
    ("tunnel",): _TUNNEL,
    ("status",): _STATUS,
    ("assess",): _ASSESS,
    ("benchmark",): _BENCHMARK,
    ("init",): _INIT,
    ("backend",): _BACKEND,
    ("models",): _MODELS,
    ("whoami",): _WHOAMI,
    ("learn",): _LEARN,
    ("explain",): _EXPLAIN,
    ("overview",): _OVERVIEW,
    ("doctor",): _DOCTOR,
    ("embeddings",): _EMBEDDINGS,
    ("embedding",): _EMBEDDINGS,
    ("rerank",): _RERANK,
    ("score",): _SCORE,
}
