# Build Plan — lobes exposes the full Colleague runtime stack as six discoverable, role-based lobes — cortex (Qwen 3.6 27B NVFP4 MTP @128K), senses (Gemma 4 12B @32K), stt, tts, embedder, reranker — that Colleague can discover, resolve to endpoints, start/stop, health-check, and measure through a stable machine-readable contract, without hardcoding any single model endpoint.

slug: `lobes-exposes-the-full-colleague-runtime-stack-as` · status: `exported` · from frame: `lobes-exposes-the-full-colleague-runtime-stack-as`

> lobes exposes the full Colleague runtime stack as six discoverable, role-based lobes — cortex (Qwen 3.6 27B NVFP4 MTP @128K), senses (Gemma 4 12B @32K), stt, tts, embedder, reranker — that Colleague can discover, resolve to endpoints, start/stop, health-check, and measure through a stable machine-readable contract, without hardcoding any single model endpoint.

## Tasks

### t1 — Role vocabulary + resolution layer: add cortex/senses to catalog.TIER_ROLE and its two mirrors (_pressure_policy._TIER_ROLE/_ROLE_TO_TIER, _tier_request); cortex->primary, senses->multimodal; pressure degrades cortex->minor; keep main|multimodal|hard|normal back-compat.

- covers: c16, h3, c4, h11
- acceptance:
  - POST model=cortex routes to the primary backend and model=senses to the multimodal backend; main|multimodal|hard|normal resolve unchanged (existing tier tests stay green)
  - under degraded pressure the ceiling maps cortex->minor; senses is NOT treated as a downgrade rung

### t2 — Generate-lane context rebalance in the fleet template: PRIMARY_MAX_MODEL_LEN 65536->131072 (cortex 128K); Gemma gear max-model-len ->32768 (senses 32K); retune PRIMARY_GPU_MEM_UTIL/MULTIMODAL_GPU_MEM_UTIL; update budget comments.

- covers: c15, h2
- acceptance:
  - fleet env.example sets PRIMARY_MAX_MODEL_LEN=131072 and the Gemma gear max-model-len=32768; the budget comment sums default utils <= ~0.70
  - template/doc-alignment tests pass (e.g. test_gateway_fleet_doc)

### t3 — Fix FLEET_CONTAINERS senses visibility: add a FLEET_MULTIMODAL constant and include the Gemma container so lobes fleet status reports its loaded/health state.

- covers: h10
- acceptance:
  - _compose.fleet_containers() includes model-gear-vllm-multimodal; lobes fleet status reports the senses container's loaded/health state

### t4 — Role registry + capability metadata core: a module defining the six roles with responsibilities/forbidden_responsibilities and role->backend/endpoint/context/quant/MTP resolution from live config; unconfigured roles report loaded=false.

- depends on: t1
- covers: c3, h10, c17
- acceptance:
  - the registry returns each of the six roles with {role, model, runtime, endpoint, context, quant, mtp, responsibilities, forbidden_responsibilities, loaded_state}
  - an unconfigured/opt-in role (e.g. stt when audio isn't up) is present with loaded=false, not absent or erroring

### t5 — CLI discovery verbs: 'lobes capabilities' (all six roles, --json) and 'lobes endpoint <role>' (base URL), read-only, consuming the role registry.

- depends on: t4
- covers: c2, h9, h4
- acceptance:
  - lobes capabilities --json lists all six roles each with the required metadata fields; lobes endpoint cortex prints the base URL
  - both verbs are read-only (no compose/deploy mutation) and read live state, nothing hardcoded

### t6 — Gateway GET /capabilities endpoint emitting the #81-shape role->endpoint JSON contract, reusing the gateway /status plumbing + the role registry.

- depends on: t4
- covers: c20, h7, c12
- acceptance:
  - GET /capabilities returns JSON where each role carries {endpoint, model, context, ready, responsibilities}; a test asserts the documented shape round-trips
  - zero hardcoded model ids in the payload; unconfigured roles report ready=false

### t8 — Per-role runtime measurement (read-only verb): LLM roles TTFT/decode-TPS/prefill/ctx/mem/restart; STT/TTS in-out duration+latency+RTF+failure; embed/rerank docs-sec/batch/latency/loaded.

- depends on: t4, t5
- covers: c19, h6, c7, h14
- acceptance:
  - each role emits its role-appropriate metric set via a read-only verb that never mutates the deployment
  - no emitted field asserts answer correctness or agent-task success (runtime-only)

### t9 — Comparison benchmark profiles: cortex-only vs cortex+senses vs senses-direct vs Qwen NVFP4-vs-BF16 (where both catalog-present); emits comparable RUNTIME metrics only.

- depends on: t8
- covers: c21, h8, c6, h13
- acceptance:
  - a profile mode runs cortex-only and cortex+senses and emits side-by-side runtime metrics; NVFP4-vs-BF16 when both are catalog-present
  - output is runtime-only; it asserts no task-quality/agent-behavior numbers

### t11 — End-to-end Colleague-contract test: a zero-model-id client resolves cortex+senses (+service roles) purely by role from the contract and drives them; asserts runtime-only.

- depends on: t5, t6
- covers: h1, h15
- acceptance:
  - a test drives cortex+senses via role-resolved endpoints with zero hardcoded model ids and passes
  - the test asserts lobes emits only runtime metrics, never task-quality claims

### t7 — Role-based serving: 'lobes up <role>' (start/stop/check one role; dry-run + --apply) and a 'colleague-stack' compose profile bringing up the six-role combination, wired to the gateway via an optional-backend pair.

- depends on: t1, t2, t5, t8
- covers: c18, h5, c14, h17
- acceptance:
  - lobes up colleague-stack --apply leaves cortex+senses+embedder+reranker (+stt/tts when audio) reachable; dry-run by default, requires --apply
  - lobes up <role> toggles a single role without disturbing the others; colleague-stack is a real compose profile

### t10 — Docs + explain entry: document the six-role contract, the JSON shape, the cortex/senses<->primary/multimodal mapping, and the before->after migration (incl. the stale 256K note); add a 'lobes explain' role-contract entry.

- depends on: t4, t5, t6, t7
- covers: c5, h12, c1
- acceptance:
  - docs describe the six roles + JSON contract shape + cortex/senses->primary/multimodal mapping + the 64K/128K -> 128K/32K migration; doc-test-alignment passes
  - a 'lobes explain' entry for the role contract exists and renders

### t12 — Live GB10 co-residence validation: deploy the rebalanced duo; confirm cortex /health@128K AND senses /health@32K simultaneously within budget; record measured utils back into the template defaults.

- depends on: t2, t7
- covers: c13, h16
- acceptance:
  - a live deploy shows cortex /health OK at 128K and senses /health OK at 32K at the same time, summed utils within budget
  - the measured PRIMARY_GPU_MEM_UTIL/MULTIMODAL_GPU_MEM_UTIL are recorded back into the fleet template defaults

## Risks

- [unknown_nonblocking] Exact retuned GPU utils for cortex@128K/senses@32K are validated live (t12); the fleet template util defaults are provisional until that run records measured values. (task t12)
- [unknown_nonblocking] Canonical responsibilities/forbidden_responsibilities word lists are finalized at implementation of t4 (issue #81 gives worked examples; final wording is a build-time call). (task t4)
- [unknown_nonblocking] 'lobes up <role>' verb surface — new top-level verb vs extending 'lobes fleet up <role>' vs 'lobes serve <role>' — resolved when building t7; behavior (one role, dry-run+--apply) is fixed either way. (task t7)
- [unknown_nonblocking] Whether the colleague-stack profile bundles the audio overlay (stt/tts) by default or assumes a separate 'lobes init --fleet --audio' — decided in t7; affects whether colleague-stack alone yields six roles or four+two. (task t7)
- [follow_up] lobes/cli/__init__.py verb-registration is a shared-write hotspot across t5/t7/t8 and lobes/templates/fleet/docker-compose.yml across t2/t7 — serialized via explicit deps so parallel waves don't collide at merge (waves narrow accordingly). (task t7)
- [unknown_nonblocking] t12 (live GB10 co-residence validation) requires physical DGX Spark hardware; it is an operational verify step, not a mergeable code change, so it can't be fanned out to a code-only workforce agent. (task t12)

## Waves (deterministic dependency schedule)

Emitted by `devague plan waves`. Each wave is file-disjoint (verified by hand,
not just by the dep graph): the `cli/__init__.py` verb-registration editors
(t5, t8, t7) are separated across waves 3/4/5, and the `docker-compose.yml` /
`env.example` writers (t2, t7, t12) are dependency-ordered.

| Wave | Tasks | Notes |
|------|-------|-------|
| 1 | t1, t2, t3 | Foundations — disjoint files (gateway/catalog · templates · `_compose.py`). |
| 2 | t4 | Role registry (depends on t1). |
| 3 | t5, t6 | Discovery CLI · gateway endpoint (disjoint: `cli/` vs `gateway/server.py`). |
| 4 | t8, t11 | Measurement verb · e2e test (new test file, disjoint). |
| 5 | t7, t9 | Role serving · bench profiles (disjoint: `cli/_commands/up.py` vs `bench/`). |
| 6 | t10, t12 | Docs/explain · **live GB10 validation**. |

**Workforce note:** waves 1–5 are code tasks fannable to `/assign-to-workforce`
(isolated worktrees, TDD-gated merges). **t12 (wave 6) is hardware-bound** —
run it on the DGX Spark yourself; it can't be delegated to a code-only agent.
Per the repo's mutation-safety rule, the write-verb tasks (t7 serving, t2/t12
template edits) default to dry-run and require `--apply`.
