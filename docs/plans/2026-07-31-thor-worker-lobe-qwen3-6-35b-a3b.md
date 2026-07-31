# Build Plan — thor worker lobe (Qwen3.6-35B-A3B)

slug: `thor-worker-lobe-qwen3-6-35b-a3b` · status: `exported` · from frame: `thor-worker-lobe-qwen3-6-35b-a3b`

> Thor moves off the Gemma 4 31B muse and hosts unsloth/Qwen3.6-35B-A3B-NVFP4 as 'worker' — a new fast ground-work lobe with self-draft MTP

## Tasks

### t1 — Catalog + tier vocabulary: add the unsloth/Qwen3.6-35B-A3B-NVFP4 entry (role_hint='worker') with values read from the checkpoint's config.json + hf_quant_config.json; TIER_ROLE and the pressure-policy tier dicts gain worker at position minor < multimodal < worker < muse < main

- covers: c2, h1
- acceptance:
  - tests/test_catalog.py passes with the new entry: id, native_max_model_len, quantization, speculative_config all match the checkpoint's actual config files (fetched at implementation time, cited in the entry comment) — card prose is not the source
  - TIER_ROLE and _pressure_policy's tier dicts both gain worker in order minor < multimodal < worker < muse < main, and a test asserts the two dicts stay identical
  - runtime._parser.infer_parser resolves the unsloth id to qwen3_coder and the catalog pairing guard passes

### t2 — Role registry: eighth role in lobes/roles.py (ROLES, ROLE_BACKEND, ROLE_ROLE_HINT, ROLE_PATH, ROLE_RESPONSIBILITIES, ROLE_FORBIDDEN, ROLE_MAX_MODEL_LEN_ENV) + profiles/schema.py ROLES gains worker; worker's doer contract encoded

- depends on: t1
- covers: c3, h2, c13
- acceptance:
  - build_role_registry returns exactly eight roles with worker present (loaded=False when unwired); a test asserts every ROLE_* dict carries a worker entry — no stale seven-role literal in code or tests
  - ROLE_RESPONSIBILITIES['worker'] includes ground-work execution tokens + tool_use + repo_action; ROLE_FORBIDDEN['worker'] == ('final_decision','security_decision')
  - profiles/schema.py ROLES includes worker; unknown-role override rejection behaviour unchanged

### t3 — Gateway opt-in wiring: _config.py gains the WORKER_BASE_URL-gated backend, worker joins OPT_IN_BACKENDS (unwired → infeasible by default), and WORKER_* keys land in FEASIBLE_ENV / PEER_ORIGIN_ENV / PEER_PROXY_ENV / PEER_API_KEY_ENV

- depends on: t1
- covers: c4, h3
- acceptance:
  - with no WORKER_* env set, the worker backend is unwired AND infeasible: gateway tests prove model=worker 404s role_infeasible, never a silent fallback
  - WORKER_BASE_URL wires the backend; WORKER_FEASIBLE / WORKER_PEER_ORIGIN / WORKER_PEER_PROXY / WORKER_PEER_API_KEY ride the same four channels — tests mirror muse's coverage

### t4 — Fleet template + activation env: profile-gated vllm-worker service on the Qwen-lane image with the qwen3_coder + qwen3 parser pair, WORKER_* knobs in env.example, shape_render.OPT_IN_CORE_ACTIVATION_ENV maps worker

- depends on: t1, t3
- covers: c5, c7
- acceptance:
  - docker compose config with COMPOSE_PROFILES=worker resolves vllm-worker on the vllm-openai nightly pin (NOT the gemma4 image) carrying --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 and WORKER_* substitutions; without the profile the service is absent
  - OPT_IN_CORE_ACTIVATION_ENV['worker'] renders COMPOSE_PROFILES gate + WORKER_BASE_URL=<http://vllm-worker:8000>; env.example documents every WORKER_* knob

### t5 — Shape machinery + honesty defaults: shapes.OPT_IN_CORE_ROLES += 'worker', base.toml vetoes worker on unrecognised cards, machine-as-brain/spark/thor/base goldens prove byte-identical rendering

- depends on: t2
- covers: c9, h11
- acceptance:
  - OPT_IN_CORE_ROLES == ('muse','worker'); DEFAULT_HOSTED_ROLES excludes both; shape loader accepts worker in hosts and overrides
  - goldens for machine-as-brain / spark / thor / base regenerate byte-identical with no WORKER_* env set — the diff is empty
  - base.toml [roles.worker] feasible=false; a base-profile render emits the veto exactly as it does for muse

### t6 — CLI verbs: lobes up worker → vllm-worker (muse-style error when unwired), capabilities / status / overview / measure / init name the eighth role, help text updated

- depends on: t2, t3
- covers: c15
- acceptance:
  - lobes up worker maps to vllm-worker and errors helpfully when COMPOSE_PROFILES lacks worker (mirrors the muse error path, tested)
  - capabilities/status/overview/measure output and the up-verb help text list worker — the audience discovers worker from CLI/gateway surfaces without reading source

### t7 — LIVE measurement on the physical Thor + thor-worker shape commit: boot the checkpoint, measure gpu_mem_util/max_model_len, choose the sm_110 MoE backend, verify the MTP spec-config loads; commit thor-worker.toml with measured values + its goldens

- depends on: t4, t5
- covers: c6, h5, c10, h12, h4
- acceptance:
  - thor-worker.toml lands with hosts [worker, embedder, reranker, stt, tts] and MEASURED gpu_mem_util / max_model_len — any refused hypothesis recorded in TOML comments (the thor-muse 0.40→0.55 pattern); no committed value precedes its measurement
  - the chosen sm_110 MoE backend and the MTP verdict (loads + acceptance rate, or recorded failure with catalog speculative_config corrected) are written into the shape/catalog data from the live boot log
  - the rendered thor-worker deployment boots vllm-worker on the Qwen-lane image on the physical Thor and serves /v1/chat/completions through the gateway; thor-worker goldens regenerate and pass

### t8 — Docs: colleague-stack.md documents eight roles + the widened division of labour (worker may act; cortex-only-actor wording updated everywhere), deployment-shapes/gateway-fleet/gemma-4-31b/qwen3.6-35b-a3b docs + CLAUDE.md updated, muse marked dormant/unhosted

- depends on: t2
- covers: c14, h7
- acceptance:
  - docs/colleague-stack.md lists eight roles with worker's responsibilities/forbidden exactly as served; every 'cortex is the only lobe that acts' claim is updated
  - muse is documented dormant/unhosted (docs + CLAUDE.md); thor-muse shape and the 31B catalog entry stay in-tree; docs/qwen3.6-35b-a3b-nvfp4.md gains the unsloth variant story; doc-test-alignment spot-check passes

### t9 — Acceptance run on the deployed Thor: switch the box to thor-worker (strip muse hosting, no MUSE_PEER_ORIGIN), verify every live signal, measure decode speed, land the evidence transcript under docs/evidence/

- depends on: t6, t7, t8
- covers: c1, h10, h6, h8, c16, h14, c17, h15, c18, h16, c19, h17, h13
- acceptance:
  - transcript under docs/evidence/ records: lobes capabilities on the deployed Thor shows worker feasible+ready serving the unsloth id, muse feasible:false with NO hosted_by, and model=muse 404s role_infeasible
  - a model=worker chat completion returns a parsed tool_calls array (not content); a skip_special_tokens:false probe shows the reasoning trace separating via --reasoning-parser=qwen3
  - decode tok/s measured with MTP on (and off, if feasible) via lobes measure/benchmark — the speed rationale is measured, not assumed
  - the transcript records the pre-move state (thor-muse serving per the 2026-07-17 evidence) and the post-move state, and the deployed .env carries no muse hosting keys
  - LIVE image probe with ground truth + negative control (mirrors the senses Red/Blue vision validation): send worker an image via model=worker and verify it correctly identifies known content, AND a deliberately-wrong assertion correctly fails. Record the HONEST verdict — vision serves, OR a #101-style gap where vLLM drops the image content part for Qwen3_5MoeForConditionalGeneration; the announcement's multimodal claim degrades honestly if it does not serve
  - LIVE video probe: send worker a short clip (video_url, the card's fps extra_body knob) and verify a correct answer about its content — or record honestly that video intake is not served on this vLLM/arch
  - LIVE thinking+coding probe exercising the doer contract: a real coding task through model=worker returns a correct code answer WITH a separated reasoning/<think> trace (via --reasoning-parser=qwen3), with MTP self-draft active — recorded with the measured draft-acceptance/decode numbers, not assumed

## Risks

- [unknown_nonblocking] sm_110 MoE backend is unknown until the t7 live boot — the card's flashinfer_b12x + CUTE_DSL_ARCH=sm_121a is Spark-arch-specific; marlin is verified only GB10-solo on the mmangkad sibling (task t7)
- [unknown_nonblocking] MTP loadability on the deployed vLLM image is unverified — the mmangkad sibling's MTP failed with a weight-shape mismatch; if the unsloth module fails to load, the catalog entry records the absence and the announcement's MTP claim degrades honestly (h1) (task t7)
- [unknown_nonblocking] t7 and t9 need the PHYSICAL Thor — they cannot fan out to isolated worktree agents, and the box currently serves muse in production: boot ordering gotchas apply (drop_caches before any recreate; depends_on can orphan a service in 'created') (task t7)
- [unknown_nonblocking] the vllm-openai nightly pin's sm_110 compatibility for the MoE marlin/flashinfer kernels is unproven (cu-wheel arch lesson: dump SASS before trusting a pin) — if the pinned image lacks sm_110 MoE kernels, t7 must select or build a Thor-safe image before measuring (task t7)
