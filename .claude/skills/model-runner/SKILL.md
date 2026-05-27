---
name: model-runner
description: Run, assess, and switch the local vLLM model via the `model` CLI (model-gear). Use when changing the served model, starting/stopping the server, load-testing a candidate (e.g. an NVFP4 checkpoint), or producing the numbers for a per-model doc under docs/.
---

# model-runner

The local vLLM model is served by this repo's deployment (`docker-compose.yml` +
`.env`, scaffolded into `~/.model-gear`). The **canonical implementation** is the
`model` CLI (the `model-gear` package); this skill points a maintainer at it and
the `scripts/model-runner.sh` shim just forwards to `model`.

It is a maintainer convenience — the deployed agent (`model-gear`) does not run it.

## When to use

- Starting/stopping the server or switching which model is served.
- Load-testing a candidate before adopting it (a model card may bless a different
  engine — verify it serves under *our* vLLM image anyway).
- Producing the correctness/throughput blocks for a per-model doc (see
  [`docs/qwen3-32b-nvfp4.md`](../../../docs/qwen3-32b-nvfp4.md)).

## How to run

Use the `model` CLI directly (or `scripts/model-runner.sh <args>`, which `exec`s it):

```bash
model init --apply                 # scaffold ~/.model-gear (compose + .env)

# switch the served model (edits .env, recreates the container, waits for health).
# DRY-RUN by default: prints the plan and changes nothing. Add --apply to execute.
model switch mmangkad/Qwen3.6-27B-NVFP4 --max-model-len 32768          # preview
model switch mmangkad/Qwen3.6-27B-NVFP4 --max-model-len 32768 --apply

model serve --apply                # start the server (alias: start)
model stop --apply                 # stop + remove the container

model assess                       # correctness probes + reasoning-trace field (markdown)
model benchmark                    # decode throughput + prefill latency (markdown)
model status                       # current model + container health (read-only)
```

Write verbs (`switch`, `serve`, `stop`, `init`) are **dry-run by default** and
require `--apply` — per CLAUDE.md's mutation-safety rule. `--port` defaults to
`VLLM_PORT` in `.env` (then 8000). `assess`/`benchmark`/`status` are read-only.
`assess` and `benchmark` each emit a markdown block ready to paste into a
per-model doc; both are stdlib-only (no third-party deps).

## Notes

- Only one ~30B-class model fits on a single GB10 at a time; `switch` does
  `docker compose down` before `up`, so it frees the prior model first.
- The reasoning-trace field name varies by vLLM build (`reasoning` on nv26.04,
  `reasoning_content` on older builds); `assess` detects and reports whichever
  carried the trace.
- `.env` is git-ignored and per-machine — `switch` writes the `VLLM_*` keys
  there. Nothing this skill writes is committed.
