# Build Plan — DSpark speculation on the Spark cortex

slug: `dspark-speculation-on-the-spark-cortex` · status: `exported` · from frame: `dspark-speculation-on-the-spark-cortex`

> The Spark's cortex got faster without swapping the model or the engine: a DSpark block-speculative drafter now rides the deployed vLLM NVFP4 lane, measured against its own 19.9-24.0 tok/s baseline on the same box — and the repo records SGLang as a known engine axis, so the framework that first demonstrated this is in the history rather than lost to a forum post.

## Tasks

### t1 — Add `ENGINE_SGLANG` to the catalog's engine axis

- covers: c11, h5
- acceptance:
  - lobes/catalog.py declares `ENGINE_SGLANG` alongside `ENGINE_VLLM` and `ENGINE_LLAMA_CPP`, and the ENGINES set that tests/`test_catalog.py` pins includes it
  - `serves_with_vllm`() returns an unchanged answer for every catalog entry that exists today — pinned by a test that enumerates the current ids
  - uv run pytest tests/`test_catalog.py` passes; no lane, profile, shape or rendered .env changes

### t2 — Add the RadixArk NVFP4 target and DSpark drafter as revision-pinned candidate catalog entries

- depends on: t1
- covers: c23, h17
- acceptance:
  - Both entries carry `role_hint`=candidate so no tier alias resolves to them, and the NVFP4 target declares engine=`ENGINE_SGLANG`
  - Each entry pins an explicit HF revision (the SGLang recipe's 52d1adc5f38aa5ebf099c29ed7025ba34cfbb854 for the target), never a floating ref, and a test asserts the pin is non-empty
  - uv run pytest tests/`test_catalog.py` passes and lobes overview --list renders both without error

### t3 — Write docs/dspark-speculation.md — the SGLang recipe and the lossless framing, both marked UNVALIDATED-by-this-repo

- covers: c17, h14, c13, h11, c7, h9
- acceptance:
  - Records the full 34-38 tok/s recipe (checkpoint, revision, image, drafter, block size, flags, content-dependent 43-47 / 12-18 spread) with its NVIDIA-forum and RadixArk sources cited inline
  - Every third-party number is labelled as published-elsewhere, never as measured here; the GGUF 15-18 tok/s figures are cited the same way and the doc states the GGUF question is closed, not re-opened
  - States 'lossless by construction' as a property of block speculation (the target verifies every drafted token) explicitly NOT measured by this work
  - markdownlint-cli2 passes on the new file

### t4 — Build scripts/spike-preflight.sh — peer-state read, rendered-argv proof, stop announcement, restore verification

- covers: c18, h20, c21, h16, c14, h12, c26, h19, c12, h10
- acceptance:
  - Reads and prints every reachable peer's `PRIMARY_PEER_ORIGIN` / `PRIMARY_PEER_PROXY` / `PRIMARY_FEASIBLE` before any mutation, in a form that pastes into the transcript
  - Prints the container's ACTUAL rendered --speculative-config argv token (from docker inspect or the live command line), not the .env line, and exits non-zero if the token is absent or brace-mangled
  - Emits the operator stop-announcement block BEFORE issuing any stop, and refuses to proceed without an explicit --apply flag (repo mutation-safety convention)
  - restore mode verifies recovery with lobes status AND one live generate through the gateway, exiting non-zero if either fails
  - Run read-only against the current healthy lane, it exits 0 and mutates nothing (verified by git status + docker ps unchanged)

### t5 — Build scripts/spec-arms.py — the three-arm, multi-shape measurement runner

- covers: c10, h4, c27, h22, c9, h3
- acceptance:
  - Measures at least three content shapes (structured/code, reasoning trace, free-form prose) and labels every reported number with both the shape and the arm in force
  - Drives all three arms — mtp-n2, dspark, none — over identical prompts and harness, and reports a missing arm as MISSING rather than falling back to a figure from an earlier transcript
  - Reads acceptance rate from vLLM's own /metrics or a named log line and prints WHICH surface each figure came from
  - stdlib-only, same convention as lobes/assess.py and scripts/prefill-depth-curve.py; --max-seconds bounds any single measurement

### t6 — Run the live three-arm spike on the DGX Spark and land the evidence transcript

- depends on: t2, t4, t5
- covers: c1, h1, c3, h7, c8, h2, c16, h13, c22, h21, c24, h18
- acceptance:
  - docs/evidence/2026-08-24-spike-dspark-cortex-spark.txt exists and answers all three c16 outputs — load yes/no with exact error text, per-shape tok/s and accepted-tokens-per-step, and the memory delta from the 1.36B drafter
  - The vLLM DSpark load attempt is run and reported BEFORE any throughput claim; a failed load is recorded as a complete result, not a failed run
  - Every number names the `max_model_len` in force, and any window reduction is announced and reported with the same weight as the cortex stop
  - The lane ends the run healthy on the incumbent config, proven by lobes status and one live generate; if it does not within the bounded attempt, the run aborts, restores and reports

### t7 — Fold the measured result back into the docs and bump the version

- depends on: t6
- covers: c2, h6, c6, h8
- acceptance:
  - docs/qwen3.8-27b-nvfp4.md and docs/dspark-speculation.md carry the measured outcome, and the 19.9-24.0 baseline is cited as a DATED measurement from its named transcript wherever it appears
  - The transcript and doc updates name the box, the date, the image digest, the config diff, and an explicit 'what this does NOT establish' section
  - python3 .claude/skills/version-bump/scripts/bump.py minor run with a CHANGELOG entry; version-check CI passes against main

## Risks

- [unknown_nonblocking] The DSpark drafter may load but accept poorly because its aux-hidden-state statistics were learned against an FP8 target, not this W4A4 one — a weak result must not be mis-reported as a verdict on DSpark as a technique (frame c20). (task t6)
- [unknown_nonblocking] The 1.36B bf16 drafter may not fit alongside a cortex at `gpu_mem_util` 0.58 with a 1M KV pool, forcing a window trade-down mid-run — a served-contract change (frame c24), not a private budget dial. (task t6)
- [unknown_nonblocking] vLLM's dspark path may never have been exercised against an NVFP4 target by anyone; the published recipe is SGLang and vLLM's reference artifact targets an unquantized 8B. Expect rough edges, and budget the run for a diagnostic outcome rather than a clean number. (task t6)
- [follow_up] t1 and t2 both write lobes/catalog.py, so they are dependency-serialized rather than file-disjoint; they must not be fanned out into the same wave. (task t2)
