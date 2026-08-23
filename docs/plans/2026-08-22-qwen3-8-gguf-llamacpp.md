# Build Plan — qwen3.8-gguf-llamacpp

slug: `qwen3-8-gguf-llamacpp` · status: `exported` · from frame: `qwen3-8-gguf-llamacpp`

> lobes serves unsloth/Qwen3.8-27B-GGUF:UD-`Q4_K_M` via a llama.cpp lane as an alternative engine to the vLLM NVFP4 cortex

## Tasks

### t1 — Spike: prove llama.cpp serves Qwen3.8-27B UD-`Q4_K_M` on this Orin (`sm_87`) — scratch space, no repo files

- instruction: Scratch space only (~/scratch or /tmp), zero repo edits. hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-`Q4_K_M`.gguf (16.46 GB, 1.6 TB free). Run llama-server with --host 0.0.0.0 -ngl 99 and a modest -c first, then climb. Probe /v1/models, /health, a known-answer chat, tool calling, and inspect whether reasoning arrives as `reasoning_content` (llama.cpp --reasoning-format). Record tok/s from the server's own timings. Write the verdict — GO or NO-GO — before touching any other task.
- covers: c1, h1
- acceptance:
  - llama.cpp server boots with the 16.46 GB UD-`Q4_K_M` GGUF on this box and answers /v1/chat/completions with a correct known-answer response
  - single-stream decode tok/s and the largest working context are both measured and written down, not estimated
  - the reasoning-trace response shape is documented: whether callers get `reasoning_content`, thinking leaked into content, or it stripped (the c22 parity check)
  - a NO-GO is a valid recorded outcome that ends the plan with a written verdict — the spike must be able to fail honestly

### t2 — Pin a llama.cpp CUDA runtime image that boots under this box's JetPack csv-mode toolkit

- instruction: This box's NVIDIA container toolkit runs csv mode (JetPack): compose deploy.resources GPU requests FAIL here — use runtime: nvidia. Try ghcr.io/ggml-org/llama.cpp server-cuda first; if its CUDA/arch build does not cover `sm_87`, build from source against the local CUDA 13.2. Pin by digest, never a floating tag, and confirm layer offload in the boot log rather than trusting the flag.
- depends on: t1
- covers: c3, h3
- acceptance:
  - the exact image digest or build recipe is recorded in-tree; no floating tag
  - the container starts with runtime: nvidia and NO deploy.resources GPU request (csv-mode toolkit rejects the latter on this JetPack)
  - container logs confirm GPU offload of the model layers on `sm_87`, not CPU-only fallback

### t3 — Add an engine/runtime axis to the catalog so a llama.cpp gear is declarable

- instruction: Touch lobes/catalog.py only. Add the engine axis so llama.cpp gears coexist with vLLM ones; vLLM-only fields (`tool_parser`, `serve_extras`) must not be required of a GGUF gear. The bar is byte-identical rendering for every existing vLLM gear — run the goldens/tests to prove it, do not eyeball it.
- depends on: t1
- covers: c2, h2
- acceptance:
  - a GGUF/llama.cpp gear can be declared in lobes/catalog.py alongside the vLLM gears
  - every existing vLLM gear still resolves byte-identically through switch, fleet render, and `infer_parser` — proven by the existing goldens/tests, not by inspection
  - vLLM-only fields (`tool_parser`, `serve_extras`) are not required of, and not silently applied to, a llama.cpp gear

### t4 — Add the llama.cpp cortex service block to the fleet compose template

- instruction: Touch lobes/templates/fleet/docker-compose.yml only. Mirror the vllm-multimodal lane's exposure model: no host-published port, gateway reaches it by service name on the compose network. llama.cpp server flags only — no vLLM flags.
- depends on: t2, t3
- covers: c25, h16
- acceptance:
  - the new lane has NO host-published port and is reachable only on the compose network, matching the vllm-multimodal precedent
  - docker ps on the deployed box shows no published port for the llama.cpp container, and a probe from another mesh box reaches it only via the gateway origin
  - the lane carries llama.cpp server flags only — no vLLM flags leak into it

### t5 — Render the Orin llama.cpp cortex lane from repo data (profile + shape), never hand-edits

- instruction: Touch lobes/profiles/builtin/orin.toml, the new shape TOML, and their goldens. orin.toml:48 documents the hand-edit debt this task must not deepen — everything the lane needs (runtime: nvidia included) renders from repo data. Measure the budget on the box with senses stopped: weights + KV at target context + hand (~0.10 util) + pooling gears, against 61.3 GiB with ZERO swap.
- depends on: t4
- covers: c23, h14, c12, h4
- acceptance:
  - a fresh lobes init render on this box produces a bootable llama.cpp cortex lane with runtime: nvidia from repo data alone — zero manual compose edits needed
  - the shape hosts cortex + hand + the pooling gears and declares senses dropped on this box
  - the memory budget is MEASURED on this swapless 61.3 GiB box at the target context with senses stopped — weights + KV + hand + pooling gears — never computed
  - profile/shape goldens are updated and the existing card renders are unchanged

### t6 — Make gateway telemetry honest for a non-vLLM backend

- instruction: Touch lobes/`_metrics.py` (and its tests). The parser keys on vllm:\* Prometheus series, so a llama.cpp backend silently reads as all-zeros — which lobes overview --live and gateway /status then present as real. Either parse llama.cpp's own metrics or mark the lane telemetry-unsupported; silent zeros are the one unacceptable outcome.
- depends on: t1
- acceptance:
  - a llama.cpp backend reports either genuinely parsed metrics or an explicit telemetry-unsupported marker — never silent zeros presented as real numbers
  - a unit test covers the non-vLLM backend path in lobes/`_metrics.py`
  - lobes overview --live and gateway /status do not imply the llama.cpp lane is idle when it is busy

### t7 — Neutralize the Tegra spurious-iowait shedding so the local cortex survives a sugov flare

- instruction: The flare is real and recorded: sugov:0/sugov:4 cpufreq kthreads flicker in D state and inflate `nr_iowait` to ~59% with vmstat bi/bo at zero. The gate is lobes/gateway/`_pressure_policy.py`:166 (`LOBES_IOWAIT_DEGRADED_THRESHOLD`, default 50). The deployed .env still says 50 and the =100 fix currently lives only as a container shell-env override that any compose up reverts. Persist it, or corroborate iowait with real disk I/O (or PSI) in the policy — the latter is the repo-level fix and needs a unit test.
- covers: c21, h13
- acceptance:
  - during a spurious sugov iowait flare (high `nr_iowait`, zero disk I/O), cortex requests through this gateway still answer 200 — not 429
  - the setting survives a docker compose down/up cycle: it is persisted in .env or repo data, not a container shell-env override
  - if the fix is Tegra-aware or disk-corroborated sampling in `_pressure_policy.py`, a unit test covers the flare case

### t8 — Verify gateway routing, the assess harness, and the role-alias contract are untouched by the engine swap

- instruction: No repo edits expected — this task's job is to prove the boundaries hold. If lobes assess needs even one llama.cpp special-case, stop and report: boundary c6 is false and gets re-scoped rather than patched around. Same for the gateway: any code change means c4 was wrong.
- depends on: t5
- covers: c4, h6, c6, h7, c15, h8
- acceptance:
  - a live model=cortex request through this box's gateway reaches the llama.cpp backend and returns a well-formed OpenAI chat completion with ZERO gateway code changes
  - lobes assess runs green against the llama.cpp lane unmodified — if any probe needs a llama.cpp special-case, boundary c6 is false and gets re-scoped rather than papered over
  - mesh callers address role aliases (model=cortex/main) only; no llama.cpp-specific model id is required at the calling contract

### t9 — Staged cutover on the box, keeping the Spark cortex proxy as a live rollback path

- instruction: Operational, on the deployed box. Order matters: verify before-state, bring the llama.cpp lane up alongside the proxy, prove the .env-only rollback works, and only then stop vllm-multimodal to free its memory. Never remove senses first. Confirm the local answer by the ABSENCE of X-Lobes-Proxied-By, not by assuming.
- depends on: t5, t7, t8
- covers: c24, h15, c16, h9, c17, h10
- acceptance:
  - the before-state is verified on the deployed box at cutover start: .env shows `PRIMARY_PEER_PROXY` to the Spark and vllm-multimodal is the running senses lane
  - rollback to the Spark proxy is proven as an .env-only change (no peer-side edit) BEFORE senses is removed
  - senses/Gemma 12B is stopped only when the llama.cpp cortex needs its memory — never as step one
  - after cutover, model=cortex answers locally with NO X-Lobes-Proxied-By header, and the Gemma 12B container is down with its budget freed
  - at every intermediate step, model=cortex through this gateway answers correctly — proxied or local, never broken

### t10 — Measure the acceptance gates and land the evidence transcript

- instruction: Measure through the gateway, not the backend directly. Targets: >= 5 tok/s single-stream decode, >= 32768 context needle-probed, known-answer + tool-calling PASS. Follow the docs/evidence/ naming convention (YYYY-MM-DD-accept-<slug>-orin.txt). A missed target is recorded as missed — this transcript is the only thing that licenses a VALIDATED claim anywhere (#108).
- depends on: t9
- covers: c19, h5, c20, h12
- acceptance:
  - measured through the gateway path callers actually use: decode >= 5 tok/s single-stream, context >= 32768 served and needle-probed, known-answer and tool-calling probes PASS
  - the transcript lands under docs/evidence/ with the date-stamped naming convention before any doc claims VALIDATED (#108)
  - every number comes from the live run — no target is claimed met by theory or extrapolation, and a missed target is recorded as missed

### t11 — Operator hands-on acceptance and documentation of the lane

- instruction: Operator-owned: a real working session, not a probe. Then write docs/qwen3.8-27b-gguf-llamacpp.md recording measured numbers AND the honest feature gaps versus the vLLM cortex (no MTP, no ViT, `preserve_thinking` and strict tools unverified), and update CLAUDE.md to describe the Orin as a local-cortex host. Cite the t10 transcript for every number.
- depends on: t10
- covers: c18, h11
- acceptance:
  - the operator runs a real working session against the local cortex and judges latency and quality acceptable — probe-passing alone does not satisfy this
  - a per-model doc records the lane, its measured numbers, and the feature gaps versus the vLLM cortex (MTP, ViT, `preserve_thinking`, strict tools)
  - CLAUDE.md reflects the Orin as a local-cortex host, and nothing claims VALIDATED without the t10 transcript

## Risks

- [unknown_nonblocking] t1 may return NO-GO — llama.cpp may not serve this hybrid-Mamba Qwen3.8 GGUF acceptably on `sm_87`. Tasks t2-t11 are conditional on GO; a NO-GO ends the plan with a recorded verdict, which is a valid delivery, not a failure to hide (task t1)
- [unknown_nonblocking] concurrency behaviour is unmeasured: llama.cpp serves one model with N parallel slots and the c20 gate targets single-stream only — mesh callers issuing concurrent cortex requests may queue (frame park v3)
- [follow_up] the llama.cpp lane is expected to LOSE vLLM-lane features: self-hosted MTP speculative decoding, ViT multimodality, `preserve_thinking` (#93), and xgrammar strict tools (colleague#320) — each is a per-feature check in t1/t11, and any that matters to a caller becomes follow-up work
- [out_of_scope] removing senses from this box leaves the mesh with no vision host and dangles the Spark's senses-proxy wiring — explicitly out of this plan's scope per frame claim c14, and owned elsewhere
- [follow_up] a second cortex host has no mesh precedent: each role today has at most one hoster and peer referral names a single origin (frame assumption c10) — whether the Orin cortex serves only local callers or the referral story extends is undecided
