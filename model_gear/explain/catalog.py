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

The served model is what the **lepenseur** agent connects to over the acp
`vllm-local` provider — model-gear runs the engine; lepenseur is one consumer of
it.

## Verbs

- `model init [TARGET]` — scaffold a deployment dir (default `~/.model-gear`).
  Dry-run by default; `--apply` writes, `--force` overwrites.
- `model serve` (alias `start`) / `model stop` — start / stop the vLLM server.
  Dry-run by default; `--apply` to commit.
- `model switch <model>` — switch the served model. Dry-run by default;
  `--apply` recreates the container and waits for `/health`.
- `model status` — read-only: current model, container state, `/health`.
- `model assess` — read-only correctness probes + reasoning-trace detection.
- `model benchmark` — read-only decode throughput + prefill latency.
- `model overview` — snapshot of the tool, the served model, and the candidate
  list (`--current` / `--list` to filter).
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
- `model explain assess`
- `model explain backend`
- `model explain models`
"""

_SWITCH = """\
# model switch

`model switch <model>` changes which vLLM model is served. **Dry-run by
default** (prints the plan, changes nothing); `--apply` commits.

On `--apply` it writes five vars to the deployment `.env` (plus
`VLLM_TOOL_CALL_PARSER` when `--tool-call-parser` is given) —

```text
VLLM_MODEL, VLLM_SERVED_NAME, VLLM_PORT, VLLM_MAX_MODEL_LEN, VLLM_GPU_MEM_UTIL
```

— then recreates the container (`docker compose down && up -d`) and waits for
`/health` (the first run downloads weights, so this can take many minutes).

Flags: `--port`, `--max-model-len` (default 32768), `--served-name` (default:
the model name), `--gpu-mem-util` (default 0.6), `--tool-call-parser` (e.g.
`hermes` for Qwen3 dense, `qwen3_coder` for Qwen3-Coder/3.6; written to
`VLLM_TOOL_CALL_PARSER` only when given), `--compose-dir`, `--apply`, `--json`.
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
`--json`.
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
doc: decode throughput (`--decode-tokens`, default 512, forced over `--runs`
repetitions, batch=1 greedy) and prefill latency (a ~2K-token prompt). Reports
host-side facts (image tag, GPU memory) too. Correctness lives in
`model assess`. Supports `--json`.
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
this is the model the **lepenseur** agent consumes.

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
`VLLM_GPU_MEM_UTIL` (default 0.6) and `VLLM_MAX_MODEL_LEN` (default 32768).

## The must-match invariant

`VLLM_SERVED_NAME` in `.env` **must equal** the part after `vllm-local/` in
lepenseur's `culture.yaml` `model:` field, or the acp provider won't resolve the
model. `model doctor` checks this.
"""

_MODELS = """\
# model-gear models

Per-model notes live under `docs/` — one markdown file per model that has been
run on this hardware, holding the correctness + throughput numbers produced by
`model assess` and `model benchmark`.

- `docs/qwen3-32b-nvfp4.md` — `nvidia/Qwen3-32B-NVFP4`, the current runtime model.
- `docs/qwen3.6-27b-nvfp4.md` — `mmangkad/Qwen3.6-27B-NVFP4`, a candidate that
  load-tested slower on decode, so the 32B stays.

`model overview --list` lists these and flags which one is currently served.
"""

_WHOAMI = """\
# model whoami

The smallest identity probe. Reports model-gear's view: the `tool` + `version`,
the `machine` (hostname + GPU), the currently-`served_model` and `port` (read
from the deployment `.env`), the `container_health`, and the `agent` that
consumes the model (`lepenseur`, from `culture.yaml`). Read-only; supports
`--json`.
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
surface, capabilities, the currently-served model, and the candidate-model list
from `docs/`. `--current` shows only the served-model block; `--list` shows only
the candidate list. `model cli overview` is the parallel snapshot of the CLI
surface itself. Supports `--json` (`{"subject", "sections"}`). A stray path
argument is accepted and ignored, so `overview <path>` never hard-fails.
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

ENTRIES: dict[tuple[str, ...], str] = {
    (): _ROOT,
    ("model-gear",): _ROOT,
    ("model",): _ROOT,
    ("switch",): _SWITCH,
    ("serve",): _SERVE,
    ("stop",): _SERVE,
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
}
