---
name: model-runner
description: Switch lepenseur's local vLLM runtime model and assess/benchmark it. Use when changing the served model, load-testing a candidate (e.g. an NVFP4 checkpoint), or producing the numbers for a per-model doc under docs/.
---

# model-runner

lepenseur's runtime model is served by local vLLM via this repo's
`docker-compose.yml` + `.env`. This skill automates the two recurring chores
around that: **switching** which model is served, and **assessing** a served
model (correctness probes + throughput) so a `docs/<model>.md` can be filled
with real numbers.

It is a maintainer convenience — lepenseur the *agent* does not run it
(lepenseur only thinks and writes). It operates only on this repo's compose and
`.env`; it reaches no path outside the repo.

## When to use

- Trying a different runtime model, or load-testing a candidate before adopting
  it (the model card may recommend a different engine — verify it serves under
  *our* vLLM image anyway).
- Producing the benchmark block for a per-model doc (see
  [`docs/qwen3-32b-nvfp4.md`](../../../docs/qwen3-32b-nvfp4.md)).

## How to run

One script, `scripts/model-runner.sh`:

```bash
# switch the served model (edits .env, recreates the container, waits for health)
scripts/model-runner.sh switch mmangkad/Qwen3.6-27B-NVFP4 --port 8001 --max-model-len 32768

# assess whatever is currently served (host facts + correctness + throughput, as markdown)
scripts/model-runner.sh assess --port 8001

scripts/model-runner.sh status   # current model + container health
scripts/model-runner.sh down      # stop + remove the container
```

`assess` emits a markdown block (model, `/health`, correctness on two fixed
probes, the reasoning-trace field name, decode tok/s, prefill) ready to paste
into a per-model doc. `_assess.py` is stdlib-only (no third-party deps).

## Notes

- Only one ~30B-class model fits on a single GB10 at a time; `switch` does
  `docker compose down` before `up`, so it frees the prior model first.
- The reasoning-trace field name varies by vLLM build (`reasoning` on nv26.04,
  `reasoning_content` on older builds); `assess` detects and reports whichever
  carried the trace.
- `.env` is git-ignored and per-machine — `switch` writes the `VLLM_*` keys
  there. Nothing this skill writes is committed.
