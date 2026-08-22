# Deviation ledger — qwen3-8-gguf-llamacpp

Append-only record of every point where execution diverged from the confirmed
plan (`docs/plans/2026-08-22-qwen3-8-gguf-llamacpp.md`). Each entry is also
first-class state in `.devague/plans/qwen3-8-gguf-llamacpp.json` — this file is
the human-readable rendering, not a second source of truth.

Recorded via `devague deviate`; `devague deviate --list` is authoritative.

---

## d1 — t7 is not fan-out work; its acceptance criterion is already satisfied in main

- **Task:** t7 · **Classification:** acceptable · **Origin:** llm · **State:** approved
- **Affects:** t7, t9, t10

**What happened.** At fan-out time, t7 ("neutralize the Tegra spurious-iowait
shedding") was found to be already delivered. `lobes/profiles/builtin/orin.toml:272`
declares `LOBES_IOWAIT_DEGRADED_THRESHOLD = "100"` under `[host_env]` at **card**
level — so every shape rendered over `orin` inherits it, machine-as-brain included —
and the deployed `~/.lobes/.env` carries it. A prior workforce run landed this on
2026-08-04.

**Why the plan was wrong.** Frame claim c21 and task t7 were both written from a
stale 2026-07-17 memory describing the earlier state, when the fix existed only as
an ephemeral shell-env override on the gateway container that any `docker compose up`
reverted. The challenge pass inherited that staleness rather than re-verifying it.

**Resolution.** t7 re-scoped to verification only, folded into the t9/t10 bring-up:
confirm a spurious sugov flare still yields 200 rather than 429 once the local cortex
runs. No new code. Acceptance criterion 3 is N/A — it was conditional on choosing the
PSI/disk-corroborated sampling rewrite, which was not the fix taken.

---

## d2 — target context raised from the c20 floor (>=32768) to the full native 262144

- **Task:** t1 · **Classification:** acceptable · **Origin:** user · **State:** approved
- **Affects:** t1, t5, t9, t10

**What happened.** The operator raised the goal to the checkpoint's official 256K
window and authorized stopping other models to reach it.

**Arithmetic (from GGUF hparams, verified against the live boot).** Only 16 of 65
layers carry a per-token KV cache (`full_attention_interval=4`); those use GQA with
`head_count_kv=4`, `key_length=value_length=256` — so KV costs 64 KiB/token.

| ctx | KV | + weights 15.33 GiB | total |
|---|---|---|---|
| 32768 | 2.00 GiB | | 17.33 GiB |
| 131072 | 8.00 GiB | | 23.33 GiB |
| **262144** | **16.00 GiB** | | **31.33 GiB** |

Predicted 31.33 GiB; **measured ~33 GiB** live at 262144 — the prediction held within
~1.7 GiB of runtime overhead.

**Consequence.** senses (`vllm-multimodal`, util 0.45, ~27.6 GiB) stopped earlier than
t9's staged ordering placed it, taking free RAM from 23 to 47 GiB. The Spark's vision
proxy to this box broke at that moment. The operator had already accepted this
("don't worry about proxies, that will be solved; we give up gemma if we need", later
"Don't worry about stopping senses at all"), and frame claim c14 puts mesh senses
re-homing explicitly out of scope.

---

## d3 — t6 touched the presentation layer, beyond its "`_metrics.py` only" instruction

- **Task:** t6 · **Classification:** acceptable · **Origin:** llm · **State:** approved
- **Affects:** t6

**What happened.** t6's operator instruction scoped the work to
`lobes/_metrics.py` (and its tests). Delivery also changed
`lobes/cli/_live.py` and `lobes/gateway/server.py`.

**Why it was necessary, not scope creep.** t6's acceptance criterion 3 is
"`lobes overview --live` and gateway `/status` do not imply the llama.cpp lane is
idle when it is busy" — a statement about the *rendered* surfaces. Parsing
honestly in `_metrics.py` is useless if the renderer then prints `running: 0` for
a field the backend never reported. Additionally `single_sections` indexed
`metrics['running']` directly and would have raised `KeyError` on the
unknown-engine dict. The instruction was drawn too narrowly for its own
acceptance criterion; the criterion is the contract and it won.

**Guard retained.** The all-vLLM path stays byte-identical: `partial` is added to
the `/status` `busy` object only when true, so existing payload-equality tests
pass unchanged, and an empty scrape still parses as vLLM.

---

## d4 — throughput gate: measured decode is ~2.5 tok/s against a c20 floor of >=5

- **Task:** t1, t10 · **Classification:** needs-follow-up · **State:** pending operator decision
- **Affects:** c20, t1, t10, t11

**What happened.** The spike met every c20 gate except decode throughput.
Measured on the deployed box, `llama-bench -p 512 -n 128 -ngl 99`:

| build | fa | pp512 (t/s) | tg128 (t/s) |
|---|---|---|---|
| ggml-org 10573 (CUDA 12) | 1 | 63.10 ± 0.12 | 2.53 ± 0.00 |
| ggml-org 10573 (CUDA 12) | 0 | 61.98 ± 0.19 | 2.52 ± 0.03 |

**Diagnosis is measured, not inferred.** Six candidate causes were tested and
eliminated: long context (identical 2.57 tok/s at ctx 8192 and 262144), GPU core
clock (306 -> 612 MHz via `jetson_clocks` gave **1.00x**), memory clock (EMC
already pinned at max 3199 MHz), CPU saturation (no core above 54% during
generation), and flash attention (2.53 on vs 2.52 off). A workload that is 99%
"busy" on `GR3D_FREQ` yet completely insensitive to a 2x core-clock change is
launch-latency-bound, not throughput-bound.

**Correction recorded.** An earlier conclusion in this work — "GPU at 100% means
compute-bound, at the silicon's ceiling, not fixable by config" — was **wrong**.
`GR3D_FREQ` measures work submitted to the GPU, not ALU occupancy, and the board
was additionally capped at `MODE_30W` with the GPU pinned to its 306 MHz floor.
The clock fix was real and necessary; it simply was not the bottleneck.

**Resolution pending** the remaining diagnostic arm (NVIDIA CUDA-13 build vs the
benchmarked CUDA-12 build, on a CUDA 13.2 host) and, if that is also null, a
lower-quant comparison. The honest outcome may be that ~2.5 tok/s is this
platform's real figure for this checkpoint, in which case c20's floor is
renegotiated rather than the number massaged.
