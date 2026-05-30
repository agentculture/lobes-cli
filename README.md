# model-gear

`model-gear` is the tooling that **runs, assesses, and switches** the local,
OpenAI-compatible vLLM model the Culture mesh consumes. The binary is `model` —
`model switch`, `model assess`, `model serve`, and so on.

The served model is what the [`model-gear`](#model-gear-is-also-the-deployed-agent)
agent connects to over the `acp` `vllm-local` provider. The tool and the deployed
agent share one identity: the same model-gear runs the engine and consumes it.

Sibling to [`culture`](https://github.com/agentculture/culture) (the agent mesh),
[`daria`](https://github.com/agentculture/daria) (awareness), and
[`steward`](https://github.com/agentculture/steward) (alignment).

## Install

```bash
uv tool install model-gear
```

## Usage

```bash
model init --apply          # scaffold a deployment dir (default ~/.model-gear)
model serve --apply         # start the vLLM server (alias: start)
model switch nvidia/Qwen3-32B-NVFP4 --apply   # switch the served model
model status                # current model, container state, /health
model assess                # correctness probes (markdown for a per-model doc)
model benchmark             # decode throughput + prefill latency
model stop --apply          # stop the server

model overview              # tool snapshot + served model + candidate list
model whoami                # tool, machine, served model, container health
model explain switch        # markdown docs for a topic
model doctor                # diagnose docker / compose / .env / health
```

Every command supports `--json`. **Write verbs (`switch`, `serve`, `stop`,
`init`) are dry-run by default** and require `--apply` to commit — agents call
CLIs in loops, so safe-by-default is mandatory.

## Running the model locally (vLLM)

`model init` scaffolds a deployment directory (default `~/.model-gear`) from the
packaged templates: a `docker-compose.yml` that stands up the vLLM model as an
OpenAI-compatible server on `:8000`, plus a `.env`. Tuned for DGX Spark (GB10
Grace Blackwell, 128 GB unified memory) per
[build.nvidia.com/spark/vllm](https://build.nvidia.com/spark/vllm).

Prerequisites: the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
and `docker login nvcr.io` with an [NGC API key](https://org.ngc.nvidia.com/setup/api-key)
to pull the `nvcr.io/nvidia/vllm` image.

```bash
model init --apply          # writes ~/.model-gear/{docker-compose.yml,.env}
# edit ~/.model-gear/.env to set HF_TOKEN if the model repo is gated
model serve --apply         # first run downloads ~28 GB of weights (the 27B primary)
model status                # waits/reports until /health is up
```

Verify it is up:

```bash
curl -fsS http://localhost:8000/health
curl -s http://localhost:8000/v1/models   # what's WARM now (the served model), not the catalog
```

Tunables live in the deployment `.env` (`VLLM_MODEL`, `VLLM_GPU_MEM_UTIL`,
`VLLM_MAX_MODEL_LEN`, `HF_CACHE`, …). `VLLM_SERVED_NAME` must match the part
after `vllm-local/` in `culture.yaml` — `model doctor` checks this. `model
switch` rewrites these keys for you.

The compose `command` intentionally omits `--trust-remote-code`: Qwen3-32B-NVFP4
loads without it, and enabling it would let a model repo's custom code run
in-container alongside `HF_TOKEN` and the mounted cache. Add it back only for a
model whose repo ships custom modeling code. If vLLM rejects the `nvidia/`
ModelOpt checkpoint, set `VLLM_MODEL` to the vLLM-native `RedHatAI/Qwen3-32B-NVFP4`
and drop `--quantization` from the compose `command`.

## Running two models behind one gateway (fleet)

`model init --fleet` scaffolds a **three-container** deployment instead of one:
two always-warm vLLM backends (a primary + a dense fallback) and a single stdlib
**gateway** that fronts them on the host port the acp `vllm-local` provider
already expects. The gateway routes each request by its `model` field, defaults an
unknown/missing name to the primary, and fails over to the other backend if the
chosen one is down — so existing single-model clients keep working unchanged while
a second model becomes addressable by name.

```bash
model init --fleet --apply        # ~/.model-gear/{docker-compose.yml,.env,Dockerfile.gateway}
docker login nvcr.io              # NGC API key for the vLLM image
model fleet up --apply            # builds the gateway image + starts all three
model fleet status                # container states + gateway /health + /v1/models
```

```bash
curl -s http://localhost:8000/v1/models       # the two WARM backends (not the full catalog — see below)
# route explicitly by name; an unknown/missing model falls back to the primary
curl -s http://localhost:8000/v1/chat/completions -d '{"model":"RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4","messages":[...]}'
```

Both models stay loaded, so set `PRIMARY_GPU_MEM_UTIL` + `FALLBACK_GPU_MEM_UTIL`
in the fleet `.env` to sum well under 1.0 (they share the 128 GB unified memory).
`model switch` is single-model only — change fleet models by editing the fleet
`.env` and re-running `model fleet up --apply`. See `model explain fleet` /
`model explain gateway` for the routing and failover semantics, and
[`docs/gateway-fleet.md`](docs/gateway-fleet.md) for the full topology.

### Per-model notes

Each runtime model has a doc under `docs/` recording how to run it, live test
results, and caveats:

- [`docs/qwen3.6-27b-nvfp4.md`](docs/qwen3.6-27b-nvfp4.md) — the **current**
  runtime model and fleet default primary (`mmangkad/Qwen3.6-27B-NVFP4`), since
  0.10.0; load-tested on DGX Spark (~8 tok/s decode, ~7 min warm-up).
- [`docs/qwen3-32b-nvfp4.md`](docs/qwen3-32b-nvfp4.md) — the dense **candidate**
  (`nvidia/Qwen3-32B-NVFP4`), faster on decode (~9.7 tok/s); swap in via
  `PRIMARY_MODEL` / `model switch` when throughput matters more than context/vision.
- [`docs/mistral-small-3.2-24b-nvfp4.md`](docs/mistral-small-3.2-24b-nvfp4.md) —
  the dense **fallback** (`RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4`) the
  gateway fleet pairs with the primary, since 0.11.0. Load-tested 2026-05-30:
  loads reliably (~15 GiB, ~14.9 tok/s decode), text + tool calls (serve with the
  mistral tokenizer + images disabled). Replaced the 35B MoE that never loaded.
- [`docs/qwen3.6-35b-a3b-nvfp4.md`](docs/qwen3.6-35b-a3b-nvfp4.md) — the former
  **MoE fallback** (`mmangkad/Qwen3.6-35B-A3B-NVFP4`), now a candidate. It does
  **not** load reliably on a GB10 shared with other services, and two ~30B models
  do not co-reside there — see [`docs/gateway-fleet.md`](docs/gateway-fleet.md).

The numbers in each doc come from `model switch <model> --apply` then `model
assess` (correctness) and `model benchmark` (throughput). `model overview --list`
lists the catalog (these models) and flags which one is currently served.

### What's loaded vs. what's supported

Two questions that look alike but aren't:

- **What's supported (what can I warm up)?** — the curated catalog of "gears"
  model-gear knows how to serve, each tagged `load-tested` (proven on this box) or
  `configured` (declared, not yet proven). It's **static** — defined in
  `model_gear/catalog.py`, shipped in the wheel, unchanged by what's running.
  Read it with `model overview --list` or the gateway's `GET /v1/models/supported`.
- **What's loaded right now?** — the model(s) actually in GPU memory this instant
  (one in single-model mode, two in the fleet). The live source is `GET /v1/models`
  (OpenAI-standard); `model fleet status` queries it. `model status` /
  `model whoami` instead report the model the deployment is *configured* to serve
  (from `.env`) plus container health — normally the same model, but it's
  configuration (which can be stale), not a live query.

| Question | CLI | HTTP |
|---|---|---|
| What *can* I run? (catalog) | `model overview --list` | `GET /v1/models/supported` |
| What's *loaded* right now? | `model fleet status` | `GET /v1/models` |
| What's the deployment *set* to serve? | `model status` / `model whoami` | — |

Mnemonic: the catalog is *what's on the menu (and which dishes we've cooked)*;
`/v1/models` is *what's hot now*. See
[`docs/gateway-fleet.md`](docs/gateway-fleet.md#supported-catalog-vs-warm-backends).

## model-gear is also the deployed agent

`model-gear` is one identity, not two: it is the repo/tool that serves the model
*and* the local thinking agent deployed on it. The agent's runtime identity lives
in `AGENTS.md` (the `acp` system prompt) and `culture.yaml` (`suffix: model-gear`,
`backend: acp`, `model: vllm-local/mmangkad/Qwen3.6-27B-NVFP4`) — the same
model-gear that runs the engine consumes it over the `acp` `vllm-local` provider.
