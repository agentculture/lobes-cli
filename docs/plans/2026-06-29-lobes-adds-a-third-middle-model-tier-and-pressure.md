# Build Plan — lobes adds a third (middle) model tier and pressure-aware tier routing

slug: `lobes-adds-a-third-middle-model-tier-and-pressure` · status: `exported` · from frame: `lobes-adds-a-third-middle-model-tier-and-pressure`

> Three capability tiers — **cheap** (4B bf16) / **normal** (14B NVFP4) / **hard** (27B @ 128K) — stay co-resident on the Spark behind the gateway. Callers request a tier *alias* (`model=cheap|normal|hard`) instead of a model name; the 27B context is trimmed 256K→128K to make room; a full swap/iowait pressure policy governs tier resolution. LoRA training stays on the existing 4B bf16 `minor` (the 14B is inference-only — *not* a LoRA base). Implements [issue #68](https://github.com/agentculture/lobes-cli/issues/68).

## Tasks

### t1 — Add the 14B-class NVFP4 middle gear to the catalog with a tier->role mapping

- covers: c4, h3
- acceptance:
  - catalog.py defines a generate gear with role_hint='middle' (the 14B NVFP4 checkpoint) plus a tier map cheap->minor / normal->middle / hard->primary; tool_parser==infer_parser(id) and quantization is non-empty per the catalog's generate-gear rules
  - tests assert the middle gear exists, is task=generate, and the tier map resolves cheap/normal/hard to the 4B/14B/27B gears

### t2 — Add a read-only host memory-pressure sampler

- covers: c17, h7
- acceptance:
  - lobes/runtime/_pressure.py computes swap_used_percent from /proc/meminfo (SwapTotal/SwapFree) and iowait_percent from /proc/stat, with no writes and no container restarts
  - unit tests feed fixture /proc samples and assert the computed swap%/iowait%, and assert the sampler opens nothing for writing (side-effect-free)

### t3 — Add the opt-in vllm-middle fleet service and trim the 27B default context to 128K

- depends on: t1
- covers: c5, c15, h5, h11
- acceptance:
  - fleet docker-compose.yml gains an opt-in vllm-middle service wired to MIDDLE_BASE_URL/MIDDLE_SERVED_NAME/MIDDLE_GPU_MEM_UTIL/MIDDLE_MAX_MODEL_LEN; PRIMARY_MAX_MODEL_LEN default changes 262144 -> 131072
  - env.example documents the MIDDLE_* vars and shows the GPU-mem-util budget summing under 1.0 (primary@128K + middle + minor + embed + rerank)

### t4 — Add the pressure policy and degraded-mode state machine (pure logic)

- depends on: t2
- covers: c18, h8
- acceptance:
  - lobes/gateway/_pressure_policy.py maps (swap%, iowait%) -> {mode, max_allowed_tier, reason} via config-driven thresholds matching #68 (swap 50/65/75, iowait 25/50 sustained); thresholds are named constants/env, not inline magic numbers
  - unit tests cover each #68 threshold band and assert the resulting mode + max allowed tier + reason

### t5 — Add the gateway tier-alias layer and wire the middle backend

- depends on: t1, t3
- covers: c1, c2, c5, c16, c21, h2, h6, h12, h14
- acceptance:
  - the gateway resolves model=cheap|normal|hard to the 4B/14B/27B served names as a same-task (generate) alias on top of task-family routing; an embed request still never fails over to a generate backend
  - tests assert model=normal routes to the 14B served name, all three aliases resolve, and the middle backend is wired only when MIDDLE_* env is present

### t6 — Add pressure-aware downgrade and manual-override handling at the gateway

- depends on: t4, t5
- covers: c18, c19, h8, h9
- acceptance:
  - under simulated swap>75% a model=hard request is served by the cheap tier; the response carries reason=pressure (HTTP header) and the response model field reflects the actually-served model
  - with the override header set, model=hard is served by the 27B even under simulated degraded pressure; tests cover both the downgrade and the override path

### t7 — Add 'lobes status --pressure' and a no-train boundary guard

- depends on: t2, t4
- covers: c20, h10, h15
- acceptance:
  - lobes status --pressure --json emits exactly {tier, model, mode, reason, pressure{swap_used_percent, iowait_percent}} and mutates nothing (read-only, like the existing status verb)
  - a test asserts the CLI registers no train/fine-tune verb (boundary guard for the LoRA-training non-goal)

### t8 — Document the three-tier fleet, alias usage, memory budget, and scope

- depends on: t1, t3, t5
- covers: c3, c9
- acceptance:
  - docs add a middle-model doc (docs/qwen3-14b-nvfp4.md), update docs/gateway-fleet.md with the tier table + alias usage + memory-budget math, and CLAUDE.md reflects the third tier
  - docs state explicitly that LoRA training stays on the 4B bf16 minor and there is no lobes train code path, with a caller-migration example from a hardcoded model id to a tier alias

### t9 — Live-validate the three-tier fleet on the Spark

- depends on: t3, t5, t6, t7
- covers: c1, c15, h1, h5, h13
- acceptance:
  - on the Spark: lobes fleet status shows all five gears healthy and nvidia-smi confirms the memory budget fits 128GB; model=normal is served by the 14B; simulated swap>75% downgrades hard->cheap with reason=pressure; the override header forces the 27B
  - measured swap/iowait deltas are recorded showing routine work on the 4B/14B lowers pressure versus 27B-as-default

## Risks

- [unknown_nonblocking] Exact 14B NVFP4 HF checkpoint must be chosen and load-tested (gibberish risk on the Blackwell vLLM image). Proceed with a dense Qwen3-14B-NVFP4 candidate; verify before promoting. (task t1)
- [unknown_nonblocking] Exact HTTP header names for the tier override + downgrade reason, and whether a streaming response can carry the reason (header vs first-chunk metadata), are unresolved. (task t6)
- [out_of_scope] t9 requires the live DGX Spark + GPUs + a running fleet; it cannot run in an isolated worktree fan-out and must be operator-run, not assigned to a workforce subagent. (task t9)
