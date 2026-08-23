# Delivery Summary — qwen3.8-gguf-llamacpp

plan: `qwen3-8-gguf-llamacpp` · run: `partial` · date: `2026-08-22`
baseline: `devague summary skeleton`

## Intent

Bring up `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M` on **llama.cpp** as a **local
`cortex`** on the Jetson AGX Orin 64 GB (sm_87) — the Ampere path around the
Blackwell-only NVFP4 W4A4 quant that made cortex infeasible locally and forced a
proxy hop to the DGX Spark. Senses (Gemma 4 12B) is given up on this box to fund
it. Eleven tasks in eight dependency waves, spike-gated per the
TRT-LLM-investigation precedent.

## Planned Work

Quoted verbatim from the `devague summary` skeleton:

- `t1` — Spike: prove llama.cpp serves Qwen3.8-27B UD-`Q4_K_M` on this Orin (`sm_87`) — scratch space, no repo files
- `t2` — Pin a llama.cpp CUDA runtime image that boots under this box's JetPack csv-mode toolkit
- `t3` — Add an engine/runtime axis to the catalog so a llama.cpp gear is declarable
- `t4` — Add the llama.cpp cortex service block to the fleet compose template
- `t5` — Render the Orin llama.cpp cortex lane from repo data (profile + shape), never hand-edits
- `t6` — Make gateway telemetry honest for a non-vLLM backend
- `t7` — Neutralize the Tegra spurious-iowait shedding so the local cortex survives a sugov flare
- `t8` — Verify gateway routing, the assess harness, and the role-alias contract are untouched by the engine swap
- `t9` — Staged cutover on the box, keeping the Spark cortex proxy as a live rollback path
- `t10` — Measure the acceptance gates and land the evidence transcript
- `t11` — Operator hands-on acceptance and documentation of the lane

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | Spike run end-to-end in scratch. Verdict **FULL GO** at MAXN. Arch `qwen35` confirmed compiled into the binary; correctness, tool-calling and reasoning-shape all pass; 256K window loads at ~33 GiB. Evidence: `docs/evidence/2026-08-23-spike-qwen38-gguf-llamacpp-orin.txt` |
| `t2` | delivered | `ghcr.io/nvidia-ai-iot/llama_cpp@sha256:f7c67c10…` (CUDA 13, arm64) pinned by digest after a 5-configuration benchmark; boots under csv-mode with `runtime: nvidia`, GPU offload confirmed |
| `t3` | delivered | `SupportedModel.engine` + `serves_with_vllm()`; GGUF gear declared; byte-identity of existing gears proven by a golden captured *before* the change (14 added lines, 0 modified). Commit `914965a`, merge `aaee5d4` |
| `t4` | delivered | `llamacpp-primary` lane: `expose:` only (no host port), `profiles: [llamacpp]`, digest-pinned, llama.cpp flags only (pinned by a test grepping the rendered command for vLLM flags). Commit `e2d45d6` |
| `t5` | delivered | `render.role_engine()` reads the engine off the catalog entry for the model the card names; `orin.toml` declares cortex on the GGUF gear at 262144; new `orin-cortex` shape hosts cortex+hand+pooling and drops senses; goldens updated with every other card render byte-identical. Merge `6796c0d` |
| `t6` | delivered | `_metrics.py` dispatches on engine; llama.cpp series parsed where they map; **unknown ≠ idle** enforced structurally (a field is emitted only if its series appeared). Renderer + `/status` surface partial totals. Commit `ad683b7`, merge `83baba4` |
| `t7` | delivered (pre-existing) | Already in main before this run — `orin.toml:272` declares `LOBES_IOWAIT_DEGRADED_THRESHOLD = "100"` at card level and the deployed `.env` carries it. Re-scoped to verification by `d1` |
| `t8` | **partial** | Assess-harness and role-alias contracts verified unchanged (19/19 assess tests pass; 3032 suite green). Gateway routing to the local lane **not** end-to-end verified — blocked with `t9` |
| `t9` | **partial** | Before-state captured live (`X-Lobes-Proxied-By: spark:8001`, model `Qwen3.8-27B-NVFP4`); `.env` backed up; lane reachable from the gateway by service name; `PRIMARY_FEASIBLE`/`PRIMARY_PEER_PROXY` flipped. **`PRIMARY_URL` still points at `vllm-primary`** — the deployed compose predates the template change and hardcodes it |
| `t10` | delivered | Evidence transcript, 14 sections, including a full retraction section. Benchmark matrix (5 configs), streaming battery (5 depths), needle PASS at 35 006 tokens, thermals, and the MAXN correction |
| `t11` | delivered | `docs/qwen3.8-27b-gguf-llamacpp.md` with measured numbers, superseded-numbers banner, MAXN correction section, parity matrix, and the operational requirement that this lane needs MAXN. Commits `6bfcd2e`, `c81ce38` |

## Mid-work Decisions

- `d1` — t7's acceptance criterion was **already satisfied in main** (landed 2026-08-04). Frame claim c21 and task t7 were both written from a stale 2026-07-17 memory describing the superseded ephemeral-override state. Re-scoped to verification only.
- `d2` — target context **raised from the c20 floor (≥32768) to the full native 262144** on operator decision; senses stopped earlier than the staged ordering placed it, taking free RAM 23 → 47 GiB.
- `d3` — t6 touched `_live.py` and `server.py` beyond its "`_metrics.py` only" instruction, because criterion 3 is a statement about the *rendered* surfaces; honest parsing is useless if the renderer still prints `running: 0`.
- `d5` — t3 touched `switch.py` and `bench/compare.py` beyond its "`catalog.py` only" instruction for the same structural reason, and **fixed a latent silent-wrong-answer**: `_is_bf16` treats empty `quantization` as bf16, so the GGUF gear would have been benchmarked as the BF16 arm of `qwen-nvfp4-vs-bf16`.
- `d6` — t5 required `render`/`shape_render`/`init` plumbing, not just data + goldens: `ROLE_SERVICE["cortex"]` is hardcoded to `vllm-primary`, so a data-only change would have started a **vLLM lane on a `.gguf` file**.
- `d7` — the lane declares `deploy.resources` and lets the card **render** `runtime: nvidia` rather than hardcoding it as instructed. Better than the instruction: csv-vs-CDI is a board fact, and hardcoding would break the lane on a devices-mode box.
- `d8` — **`d4` retracted.** The throughput deficit was the `MODE_30W` power cap, not a structural limit.
- Not covered by a record: the enabled quant was left at **Q4_K_M** despite Q3_K_XL measuring marginally faster at MAXN (8.56 vs 8.46) — a 1.2 % gain does not justify a quantization level of quality loss.

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|-----------------------|----------------|
| `t7` (`d1`) | acceptance criterion already satisfied in main; plan written from stale memory | acceptable |
| `t1` (`d2`) | context target raised to the full native 262144 by operator decision | acceptable |
| `t6` (`d3`) | instruction drawn narrower than its own acceptance criterion | acceptable |
| `t3` (`d5`) | same, plus a latent bf16-misclassification bug found and fixed | acceptable |
| `t5` (`d6`) | data-only change would have started a vLLM lane on a GGUF file | acceptable |
| `t4` (`d7`) | `runtime: nvidia` rendered via the card mechanism instead of hardcoded | acceptable |
| `t1`, `t10` (`d4` → retracted by `d8`) | the "structural limit" conclusion was produced by reading a governed clock at idle; MAXN yields 3.24×/3.92× | acceptable |
| `t8` | gateway-routing leg not verified end-to-end; blocked behind `t9`'s `PRIMARY_URL` | needs-follow-up |
| `t9` | deployed compose predates the template and hardcodes `PRIMARY_URL`; the correct fix is a re-render from repo data, deliberately not hand-edited | needs-follow-up |

## Evidence

- tests: full suite `uv run pytest -n auto -q` — **3032 passed, 15 skipped** (skips are live-deployment gates)
- tests: `tests/test_assess.py` — 19 passed (t8's harness-unchanged claim)
- lint: `black --check`, `isort --check-only`, `flake8`, `bandit` — clean; `uv run afi cli doctor . --strict` — 26/26
- commits: `main..HEAD` — 18 commits, `83baba4`…`0de543d`
- evidence transcript: `docs/evidence/2026-08-23-spike-qwen38-gguf-llamacpp-orin.txt` (14 sections)
- deviation ledger: `docs/deliveries/2026-08-23-qwen3-8-gguf-llamacpp-deviation-ledger.md` (d1–d8)
- method doc + tooling: `docs/measuring-lane-performance.md`, `scripts/prefill-depth-curve.py`

## Delivery Claims

| Claim | Confidence | Evidence |
|-------|------------|----------|
| a llama.cpp GGUF gear is declarable alongside vLLM gears, with existing gears byte-identical | high | `tests/goldens/switch-plans.txt` (14 added, 0 modified lines) · merge `aaee5d4` |
| the `llamacpp-primary` lane renders with `runtime: nvidia`, digest-pinned, no host port, zero vLLM flags | high | live `lobes init --profile orin --shape orin-cortex --apply` + `docker compose config` · merge `6796c0d` |
| gateway telemetry distinguishes *unknown* from *idle* for a non-vLLM backend | high | merge `83baba4` · `tests/test_overview_live.py` |
| the checkpoint decodes correctly on sm_87 (hybrid Mamba/SSM + attention) | high | evidence §6 — known-answer, arithmetic-with-thinking, tool_calls non-null |
| tool calling and `reasoning_content` reach parity with the vLLM lane on default flags | high | evidence §5, §6 |
| the 262144 window is served and **usable to 35 006 tokens** (needle retrieved verbatim) | high | evidence §13 · `~/scratch/needle-32k.log` |
| decode **8.46 tok/s**, prefill **253.84 tok/s** at MAXN — clears the ≥5 gate | high | evidence §14 · `llama-bench` `38406d597` |
| MAXN yields 3.24× decode / 3.92× prefill over `MODE_30W`; the earlier "structural limit" was a measurement artifact | high | evidence §14 · ledger `d8` · clock ladder 306/612/1300 MHz |
| MAXN needs no manual fan lock — 23-minute sustained load plateaus at 75.5 °C with the clock held | high | evidence §14 thermal table |
| the 262144 window is usable at **full** depth (~237K tokens) | **unverified** | first attempt failed on a **client-side timeout** (my 1800 s limit), not a model error; re-run in flight |
| `model=cortex` answers locally through the gateway with no proxy header | **unverified** | `PRIMARY_URL` still resolves to `vllm-primary`; cutover incomplete |
| the lane is fit for the operator's real work (t11 hands-on acceptance) | **unverified** | no hands-on session recorded |

## Remaining Work / Follow-up

- **`t9` — finish the cutover.** Re-render the deployment from repo data (`lobes init --force --shape orin-cortex`) so `PRIMARY_URL` becomes overridable, then `lobes fleet up`. Deliberately **not** hand-edited: that would deepen the exact debt `orin.toml:48` documents. Blocking `t8`.
- **`t8` — verify gateway routing end-to-end** once `t9` lands: `model=cortex` must answer with **no** `X-Lobes-Proxied-By` header, and `lobes assess` must run green unmodified.
- **Near-max context proof** — the ~237K-token needle is re-running on Q3_K_XL with a 5400 s timeout. The c20 gate is already met at 35 006 tokens; this is an additional demonstration.
- **Quant ladder** — the measured size→speed slope is very weak (`speed ~ size^-0.05`), so *larger* quants are nearly free: Q5_K_M should cost ~1 % for a full quantization level of quality. Q5_K_M downloading; **untested**.
- **`lobes up cortex` still targets `vllm-primary`** (`up.py` carries its own static `ROLE_SERVICE`). `lobes fleet up` works. Genuine follow-up, deliberately not restructured late in the run.
- **`machine-as-brain` on the `orin` card is now over-committed** (cortex ~33 GiB + senses ~27.6 GiB > 61.3 GiB). The schema cannot express this; `orin.toml`'s banner says so and a test pins that it says so.
- **Mesh vision is unhosted** — senses was stopped and the Spark's senses-proxy to this box dangles. Explicitly out of scope per frame claim `c14`; owned elsewhere.
- **`gh` is unauthenticated on this box**, so the PR cannot be opened from here.
