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

---

## d5 — t3 touched switch/bench beyond its "`catalog.py` only" instruction, and fixed a latent bug

- **Task:** t3 · **Classification:** acceptable · **Origin:** llm · **State:** approved
- **Affects:** t3

**What happened.** t3's instruction scoped the work to `lobes/catalog.py`. Delivery
also changed `lobes/cli/_commands/switch.py` and `lobes/bench/compare.py`.

**Why the instruction could not be satisfied as written.** t3's acceptance
criterion 3 requires that vLLM-only fields are "not silently applied" to a
llama.cpp gear. `infer_parser` matches on the model **name**, so
`unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M` resolves to `qwen3_coder`, and `lobes switch`
would have written `VLLM_TOOL_CALL_PARSER=qwen3_coder` for a gear whose engine has
no such flag — exactly the failure the engine axis exists to prevent. A
catalog-only change cannot stop that. Same shape as d3: the instruction was
narrower than its own acceptance criterion.

**Latent bug found and fixed (not in scope, but silent and wrong).**
`lobes benchmark --profile qwen-nvfp4-vs-bf16` splits its pair on `_is_bf16`,
which treats `quantization == ""` as bf16. The GGUF gear's id contains both `qwen`
and `27b`, and its `quantization` is empty for an unrelated reason — Q4_K_M lives
*inside* the GGUF file. Without a gate it would have been selected as the **BF16
arm** and benchmarked as one, producing a plausible and entirely wrong comparison.
Verified empirically before fixing; a named regression test records why.

**Byte-identity evidence.** A switch-plan golden was captured BEFORE the model
change (`tests/goldens/switch-plans.txt`, commit `833846c`) and moved by exactly
**14 added lines and zero modified lines** — criterion 2 proven by diff rather
than inspection, as the plan demanded.

---

## d6 — t5 required render/shape_render/init plumbing, not just data + goldens

- **Task:** t5 · **Classification:** acceptable · **Origin:** llm · **State:** approved

**What happened.** The plan scoped t5 to `orin.toml` + the shape TOML + goldens.
That is not sufficient: `ROLE_SERVICE["cortex"]` is hardcoded to `vllm-primary`,
so a data-only change would have started a **vLLM lane on a `.gguf` file**.

**Plumbing added, and why each piece is where it is.**

- `render.role_engine()` reads the engine off the **catalog entry for the model
  the card names** (t3's `engine` field) rather than adding a new `RoleProfile`
  knob — a knob could drift from the model it describes; a lookup cannot.

- `render.profile_env()` emits `COMPOSE_PROFILES=llamacpp` and
  `PRIMARY_URL=http://llamacpp-primary:8000` on the **profile** side, not the
  shape side. That is what keeps the identity-shape invariant
  (`machine-as-brain` == bare card profile) true on a card that declares one.

- `init._shape_dropped_services()` parks `vllm-primary` *even though cortex is
  hosted*, because it is hosted by the other engine.

- A non-vLLM model on a role with no alternative lane now **raises** in both
  layers rather than silently falling back to `vllm serve`.

---

## d7 — the lane RENDERS `runtime: nvidia` instead of hardcoding it, contrary to instruction

- **Task:** t4 · **Classification:** acceptable · **Origin:** llm · **State:** approved

**What happened.** The t4 instruction said to hardcode `runtime: nvidia` (not
`deploy.resources`) because this JetPack's container toolkit runs csv mode. The
delivered lane does the **opposite in the template** and gets the right result in
the **render**: it declares the shipped `deploy.resources` stanza and registers in
`_compose.GPU_SERVICES`; the orin card's existing `gpu_access = "runtime"` then
generates `docker-compose.gpu.yml`, which `!reset`s the stanza and sets
`runtime: nvidia`.

**Verified end-to-end**, not argued: a real `lobes init --profile orin --shape
orin-cortex --apply` followed by `docker compose config` yields `runtime: nvidia`
and `deploy: null`.

**Why this is better than the instruction.** csv-vs-CDI is a fact about a
**board**, and this repo already has the mechanism for it — the one built
precisely to stop the hand-edit debt `orin.toml:48` documents. Hardcoding
`runtime:` would bake one board's toolkit mode into a card-neutral template and
break the lane on a devices-mode box, while gaining nothing here. My instruction
was correct about the *requirement* and wrong about the *mechanism*.

---

## Follow-ups recorded, not silently dropped

- **`lobes up cortex` still targets `vllm-primary`.** `up.py` carries its own
  static `ROLE_SERVICE`, resolved before the deployment dir is known, so on an
  `orin-cortex` deployment it raises the existing "this deployment's shape drops
  that service" user error instead of starting the llama.cpp lane. `lobes fleet
  up` — the documented fleet path — works. Left alone rather than restructure a
  validated verb late in the run.

- **`machine-as-brain` on the `orin` card is now an over-committed declaration**
  (cortex ~33 GiB + senses ~27.6 GiB > 61.3 GiB, zero swap). Unavoidable given
  the schema: a shape override cannot flip `feasible`, and feasibility ("can the
  board serve it at all?") is genuinely true. `orin.toml`'s header states this in
  a banner, points at the two shapes that resolve it, and a test pins that the
  file says it.

- **`LLAMACPP_MODEL_DIR` cannot come from repo data** — it is the host directory
  holding the `.gguf`, an operator path like `HF_CACHE`. Defaults to
  `${HOME}/.cache/llama.cpp`; `env.example` documents the knob and the
  `hf download` line.

---

## d8 — **d4 IS RETRACTED**: the throughput deficit was the power cap, not a structural limit

- **Task:** t1 · **Classification:** acceptable · **Origin:** llm · **State:** approved
- **Affects:** t1, t5, t10, t11 — and **supersedes d4**

**What d4 claimed.** That measured decode of ~2.5 tok/s against the >=5 gate was
"structural ... not reachable by any configuration lever", citing seven measured
levers including a null result for GPU core clock (306 -> 612 MHz = exactly 1.00x).

**Why it was wrong.** The "306 MHz" baseline was a **measurement artifact**. The
devfreq governor (`nvhost_podgov`) was active at `min=306 / max=612`. I read
`cur_freq` **at idle**, recorded "306 MHz", and benchmarked — but under load the
governor **boosted to 612 MHz**. The comparison was therefore
*governor-boosted-612* against *pinned-612*: the same clock. It correctly returned
1.00x for a change that never happened, and that false null propagated into a
confident structural conclusion.

**How it was caught.** The operator proposed the inverted test — pin
`min=max=306` so no boost is possible — which costs a sysfs write instead of a
reboot and is decisive in either direction. Measured on the CUDA-13 build:

| GPU clock | decode |
|---|---|
| 306 MHz (pinned) | 1.36 tok/s |
| 612 MHz | 2.61 tok/s |
| **1300.5 MHz (MAXN)** | **8.46 tok/s** |

**Measured at MAXN** (`nvpmodel -m 0` + `jetson_clocks`, reboot required):

| metric | 612 MHz | MAXN | gain |
|---|---|---|---|
| pp512 prefill | 64.79 | **253.84 ± 1.45** | **3.92x** |
| tg128 decode | 2.61 | **8.46 ± 0.00** | **3.24x** |
| sustained (900 tok) | — | **8.43** | — |

The gain is **superlinear** against the 2.12x clock increase because MAXN also
restores 4 disabled CPU cores (12 online, was 8) and lifts the power budget.

**Consequences.**

- **All five c20 criteria now PASS.** The spike verdict changes from
  "functional GO / throughput FAIL-AS-SPECIFIED" to **FULL GO**.

- TTFT rule revised: `TTFT_seconds ~= depth / 254` (was `/ 57`).
- The deployment comparison changes: against the Thor's 12.1 tok/s the local lane
  is **1.4x slower, not 4.6x** — competitive, not a consolation prize.

- Evidence section 8's "launch-latency bound, not throughput bound" is withdrawn
  in section 14.

**Thermals verified at MAXN**, since the power increase is real: 100 s of
sustained generation took `tj` to 70.9 C with `nvfancontrol` auto-ramping 36% ->
52%, and the GPU **held 1300 MHz with no throttling** (~25 C of headroom). No
manual fan lock is required.

**Methodological lesson.** A governed clock read at idle does not describe the
clock under load. That single error produced a confident, well-evidenced, entirely
wrong conclusion that survived six other *correct* eliminations. Pin the frequency
(`min=max`) before attributing anything to clock.

---

## d9 — the quant recommendation was decided on a proxy, and says so

- **Task:** t11 · **Classification:** needs-follow-up · **Origin:** llm · **State:** approved

**What happened.** The lane's quant was chosen by measuring **four rungs on two
axes** — speed (`llama-bench`) and perplexity (`llama-perplexity`, wikitext-2,
200 chunks, paired) — under enforced-identical conditions.

| quant | tok/s | PPL | extra sec / 500-tok answer |
|---|---|---|---|
| UD-Q3_K_XL | 8.56 | 6.7922 | −0.7 s (but −1.20% quality) |
| **UD-Q4_K_M** | **8.46** | **6.7118** | — |
| UD-Q5_K_M | 7.15 | 6.6970 | +10.8 s for +0.22% |
| UD-Q6_K | 6.56 | 6.6857 | +17.1 s for +0.39% |

**Two predictions were refuted en route**, both recorded rather than quietly
dropped:

1. *"Going up the quant ladder is nearly free"* — extrapolated a size law from
   the Q3→Q4 pair, predicted Q5_K_M at ~8.38 tok/s, measured **7.15** (off by
   >10×). Size is not the driver; the 4-bit→5-bit **kernel** boundary is.
2. *"Q5 costs 15.5% for a quality step up"* — the cost was measured, the
   **step up was assumed**. Measured, it is 0.22%, at 0.20× the confidence
   half-width.

**Why this is `needs-follow-up` and not `acceptable`.** Perplexity is a log
measure of next-token prediction on wikitext. It is **not a decision-error
rate**, and `cortex` is the fleet's final-authority lobe — the cost of a wrong
call there is whatever it propagates into. The interim choice is defensible
because it needs no bet on the unmeasured axis (Q4 is the knee from both
directions), but the axis that matters is **unmeasured, not small**.

**Tracked in issue #194** — error rate on cortex-shaped tasks (tool-call
correctness, multi-step reasoning, instruction adherence, long-context
retrieval), with confidence intervals sized narrower than the effect claimed.
