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
model serve --apply         # first run downloads ~18 GB of weights
model status                # waits/reports until /health is up
```

Verify it is up:

```bash
curl -fsS http://localhost:8000/health
curl -s http://localhost:8000/v1/models   # lists nvidia/Qwen3-32B-NVFP4
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

### Per-model notes

Each runtime model has a doc under `docs/` recording how to run it, live test
results, and caveats:

- [`docs/qwen3-32b-nvfp4.md`](docs/qwen3-32b-nvfp4.md) — the **current** runtime
  model (`nvidia/Qwen3-32B-NVFP4`), benchmarked on DGX Spark.
- [`docs/qwen3.6-27b-nvfp4.md`](docs/qwen3.6-27b-nvfp4.md) — a **candidate**
  (`mmangkad/Qwen3.6-27B-NVFP4`), load-tested on DGX Spark; loads under the
  current vLLM image but is slower on decode, so the 32B stays.

The numbers in each doc come from `model switch <model> --apply` then `model
assess` (correctness) and `model benchmark` (throughput). `model overview --list`
lists these docs and flags which model is currently served.

## model-gear is also the deployed agent

`model-gear` is one identity, not two: it is the repo/tool that serves the model
*and* the local thinking agent deployed on it. The agent's runtime identity lives
in `AGENTS.md` (the `acp` system prompt) and `culture.yaml` (`suffix: model-gear`,
`backend: acp`, `model: vllm-local/nvidia/Qwen3-32B-NVFP4`) — the same model-gear
that runs the engine consumes it over the `acp` `vllm-local` provider.
