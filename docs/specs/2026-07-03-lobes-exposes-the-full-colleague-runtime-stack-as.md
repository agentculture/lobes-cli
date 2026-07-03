# lobes exposes the full Colleague runtime stack as six discoverable, role-based lobes — cortex (Qwen 3.6 27B NVFP4 MTP @128K), senses (Gemma 4 12B @32K), stt, tts, embedder, reranker — that Colleague can discover, resolve to endpoints, start/stop, health-check, and measure through a stable machine-readable contract, without hardcoding any single model endpoint.

> lobes exposes the full Colleague runtime stack as six discoverable, role-based lobes — cortex (Qwen 3.6 27B NVFP4 MTP @128K), senses (Gemma 4 12B @32K), stt, tts, embedder, reranker — that Colleague can discover, resolve to endpoints, start/stop, health-check, and measure through a stable machine-readable contract, without hardcoding any single model endpoint.

## Audience

- Colleague (the consuming agent runtime) is the primary client; secondary: the lobes operator running the fleet, and mesh peers. Colleague resolves roles->endpoints programmatically and must never hardcode a model id.

## Before → After

- Before: Today the generate lane is addressed only by capability-TIER aliases (main|minor|multimodal, back-compat hard|cheap|normal) with no ROLE concept; the current validated balance is INVERTED vs the target — Qwen 27B at 64K, Gemma 12B at 128K — and the four service gears (embed, rerank, and the hardcoded Parakeet STT / Chatterbox TTS) exist but are NOT discoverable as first-class named roles, with no machine-readable role->endpoint contract for Colleague.
- After: lobes exposes six first-class roles — cortex, senses, stt, tts, embedder, reranker — each resolvable by name to a live endpoint + metadata (model, runtime, context limit, quant, MTP support, responsibilities, forbidden responsibilities, health, loaded state).
- After: The generate lane is rebalanced: cortex = Qwen 3.6 27B NVFP4 MTP served at 128K; senses = Gemma 4 12B served at 32K. cortex/senses are the primary capability contract; main/multimodal and hard/normal remain as back-compat aliases.

## Why it matters

- Colleague can build a clean cognitive boundary — cortex as the authoritative reasoning/action/validation layer, senses as the user-facing intake/perception/speak-back front door — and dynamically measure whether cortex-only or Gemma-senses->Qwen-cortex is the better architecture, instead of hardcoding one monolithic model endpoint.

## Requirements

- REBALANCE: cortex (Qwen 3.6 27B NVFP4 MTP) served context 64K->128K; senses (Gemma 4 12B) served context 128K->32K; util retuned so both co-reside healthy within the GB10 budget; the swap/iowait pressure policy that degrades 'main' is redefined in cortex/senses terms (cortex->minor on pressure; senses is a distinct capability, not a rung).
  - honesty: On the fleet template PRIMARY_MAX_MODEL_LEN=131072 (cortex 128K) and the Gemma gear's max-model-len=32768 (senses 32K); a live co-resident deploy on the GB10 shows BOTH cortex and senses HEALTHY with summed GPU utils within budget (<=~0.70); and the pressure policy degrades cortex->minor (senses treated as a distinct capability, not a rung).
- ROLE NAMES: add first-class role/capability names cortex, senses, stt, tts, embedder, reranker; cortex/senses are the primary generate-lane contract; main|multimodal|hard|normal are kept as back-compat aliases that resolve to cortex/senses.
  - honesty: catalog.TIER_ROLE and its two mirrors resolve cortex->primary and senses->multimodal; POSTing model=cortex or model=senses to the gateway routes to the right backend; and model=main|multimodal|hard|normal still resolve unchanged (existing tier tests still pass).
- CAPABILITY DISCOVERY: a command/API (e.g. 'lobes capabilities', 'lobes endpoint <role>') lets a client inspect each role — role, model/alias, runtime, endpoint/base URL, context limit, quant, MTP/spec support, responsibilities, forbidden responsibilities, health status, loaded/unloaded state.
  - honesty: 'lobes capabilities' lists all six roles each with {role, model/alias, runtime, endpoint, context, quant, MTP, responsibilities, forbidden_responsibilities, health, loaded_state}, and 'lobes endpoint <role>' returns a role's base URL — both read LIVE deploy/gateway state, nothing hardcoded.
- ROLE-BASED SERVING: start/stop/check services by role — 'lobes up cortex|senses|embedder|reranker|stt|tts' and a 'colleague-stack' profile that starts the expected Colleague combination; respects the repo's dry-run-by-default --apply mutation-safety rule.
  - honesty: 'lobes up <role>' and 'lobes up colleague-stack' start/stop/check the correct container(s); colleague-stack is a real compose profile; and these WRITE verbs default to dry-run and require --apply per the repo mutation-safety rule.
- PER-ROLE MEASUREMENT: lightweight per-role metrics — LLM roles: TTFT, decode TPS, prefill, context limit, mem usage, readiness, error/restart count; STT/TTS: in/out duration, latency, real-time factor, failure rate; embedder/reranker: req/sec or docs/sec, batch size, latency, loaded state.
  - honesty: Each role reports its metric set (LLM: TTFT/decode-TPS/prefill/ctx/mem/restart; STT/TTS: in-out duration/latency/RTF/failure; embed/rerank: docs-sec/batch/latency/loaded) via a READ-ONLY verb that never mutates the deployment.
- COLLEAGUE JSON CONTRACT: a stable machine-readable output (JSON CLI mode and/or local HTTP API and/or config file) matching the #81 shape, so Colleague consumes lobes without parsing human CLI text; roles carry endpoint, model, context, ready, responsibilities.
  - honesty: The emitted JSON round-trips the #81 shape — a consumer parses each role's {endpoint, model, context, ready, responsibilities} with zero hardcoded model ids — and the schema is documented + stable (covered by a test that asserts the shape).
- MEASUREMENT PROFILES: a benchmark/profile mode comparing cortex-only vs Qwen-cortex+Gemma-senses vs Gemma-senses-direct-for-cheap-tasks vs Qwen NVFP4-vs-BF16 (where both available) — lobes measures the runtime layer for these profiles.
  - honesty: A benchmark/profile mode runs cortex-only and cortex+senses (and Qwen NVFP4-vs-BF16 where both are present) and emits comparable RUNTIME metrics only; it does not assert task-quality/agent-behavior numbers.

## Honesty conditions

- A Colleague client, given only lobes' machine-readable contract, resolves and consumes cortex+senses (plus stt/tts/embedder/reranker) endpoints BY ROLE with zero hardcoded model ids; and lobes emits only runtime metrics, never task-quality claims.
- Colleague can be pointed at lobes and resolve every role it needs BY NAME (cortex/senses/stt/tts/embedder/reranker) with no model id in its own config; the operator and mesh peers read the same contract.
- Each of the six roles is enumerable with its full metadata block; an unconfigured/opt-in role (e.g. stt/tts when audio isn't up) reports loaded=false rather than being absent or erroring.
- cortex resolves to the 27B served at 128K and senses to Gemma served at 32K; main|multimodal|hard|normal still resolve to the same backends they do today (no caller breakage).
- Verifiable today: grep shows no cortex/senses vocabulary in code; catalog.TIER_ROLE carries only main|minor|multimodal (+cheap/normal/hard); the fleet template serves primary@64K + multimodal@128K; and no role->endpoint JSON contract exists.
- With the contract in place Colleague can run the same task through cortex-only and through senses->cortex and get comparable RUNTIME numbers to choose the architecture, without editing lobes.
- Every metric lobes emits is a runtime/serving measurement; no field anywhere in the contract or bench output asserts answer correctness or agent-task success.
- A from-scratch Colleague config containing zero model ids successfully drives cortex+senses via the resolved endpoints.
- A live GB10 deploy shows cortex /health OK at 128K AND senses /health OK at 32K simultaneously, within the GPU budget.
- 'lobes up colleague-stack --apply' leaves all six roles reachable; 'lobes up <role>' toggles a single role without disturbing the others.

## Success signals

- Colleague resolves cortex and senses endpoints from lobes' machine-readable JSON with zero hardcoded model ids, and the payload matches the #81 shape ({role:{endpoint,model,context,ready,responsibilities}}).
- cortex serves 128K context and senses serves 32K context, co-resident and both HEALTHY on the 128 GB GB10 within budget.
- 'lobes up colleague-stack' (a profile) brings up the expected Colleague combination (cortex + senses + stt + tts + embedder + reranker) and 'lobes up <role>' can start/stop/check any single role.

## Scope / boundaries

- lobes measures the RUNTIME layer only (endpoint readiness, TTFT, decode TPS, prefill, RTF, docs/sec, memory, restart count); it does NOT measure task quality or agent behavior — that stays Colleague's job.

## Non-goals

- Not replacing Qwen with Gemma. cortex remains the authoritative model; senses is an additive front-door layer, not a cheaper substitute for cortex.
- 'brain' is FORBIDDEN as a role name or compatibility alias — inside the lobes metaphor every lobe is already brain-like, so the authoritative role is 'cortex', never 'brain'.
- STT (Parakeet) and TTS (Chatterbox) stay HARDCODED fixed sidecars — exposing them as roles means surfacing their existing endpoints in the contract, NOT adding them to the switchable model catalog (lobes/catalog.py).
- The existing tier aliases (main|minor|multimodal, hard|cheap|normal) are NOT removed — cortex/senses become the primary contract and the tier aliases are kept as back-compat so existing callers keep working.

## Decisions

- Each role declares first-class responsibility metadata Colleague reads: cortex responsibilities = [reasoning, deciding, planning, tool_use, code_repo_actions, validation, final_authority]; senses responsibilities = [intake, normalize_input, classify_intent, prepare_context_packet, speak_back] with forbidden_responsibilities = [final_decision, repo_action, security_decision].
- The authoritative role is named 'cortex' and the front-door role 'senses' — these supersede the tier words main/multimodal and the older cheap/normal/hard and thinker/multimodal framings as the PRIMARY contract, while those older aliases stay resolvable for back-compat.
- IMPLEMENTATION SHAPE (low blast radius): cortex/senses are a ROLE LAYER that maps onto the existing internal backend roles — cortex->primary backend, senses->multimodal backend — added to catalog.TIER_ROLE and its two mirrors (_pressure_policy._TIER_ROLE, _tier_request). The internal compose SERVICE names (vllm-primary/vllm-multimodal), env vars (PRIMARY_*/MULTIMODAL_*), and container names are NOT renamed. This preserves running deployments + the main|multimodal|hard|normal aliases while adding the richer cortex/senses role contract (endpoint + responsibilities metadata) on top.
- CONTRACT TRANSPORT: the machine-readable role contract ships as BOTH a 'lobes capabilities --json' CLI mode AND a gateway HTTP endpoint (GET /capabilities), reusing the gateway's existing /status,/health,/v1/models plumbing — the gateway is the always-warm front Colleague already talks to, so a role->endpoint HTTP contract there is the natural home; the CLI mode covers the operator/no-gateway case.
- DISCOVERABILITY PREREQUISITE: making senses a first-class reported role requires adding the Gemma/multimodal container to FLEET_CONTAINERS in runtime/_compose.py (currently omitted, so 'lobes fleet status' does not report it) — the rename/rebalance work fixes this pre-existing gap so every role's health/loaded state is reportable.

## Accepted plan risks / parked unknowns (non-blocking)

> These `unknown_nonblocking` items are retained in the frame JSON but dropped by the
> spec-md exporter; re-attached here by hand so `/spec-to-plan` sees the full picture.
> None block convergence — they are plan-time or live-measurement calls, not fabrications.

- **Exact retuned GPU-mem-utils.** `PRIMARY_GPU_MEM_UTIL` for cortex@128K (~0.30 expected to hold — the 27B KV is util-bound not context-bound, so 64K→128K just trades concurrency, same memory) and `MULTIMODAL_GPU_MEM_UTIL` for senses@32K (droppable well below the current 0.22, since Gemma-128K was already cheap at 0.22) — final values validated by a live co-resident KV measurement on the GB10.
- **Canonical responsibilities / forbidden_responsibilities word lists.** Issue #81 gives worked examples (cortex: reasoning/deciding/planning/tool_use/actions/validation/final_authority; senses: intake/normalize_input/classify_intent/prepare_context_packet/speak_back; forbidden: final_decision/repo_action/security_decision); final wording confirmed at plan time.
- **CLI shape for role-based serving.** Whether `lobes up <role>` is a NEW top-level verb, an extension of `lobes fleet up <role>`, or an alias of `lobes serve <role>` — behavior (start/stop/check one role, dry-run + `--apply`) is fixed; the exact verb surface is a plan-time call.
- **colleague-stack ↔ audio overlay.** Whether the `colleague-stack` profile bundles the audio overlay (stt/tts Parakeet/Chatterbox, currently opt-in via `lobes init --fleet --audio`) by default, or assumes audio is added separately — affects whether `colleague-stack` alone yields all six roles or four + two.

## Source

Specced from [agentculture/lobes-cli#81](https://github.com/agentculture/lobes-cli/issues/81) via the `/think` (devague) operator chain.
