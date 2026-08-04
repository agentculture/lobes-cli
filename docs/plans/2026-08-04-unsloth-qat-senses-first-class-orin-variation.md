# Build Plan — unsloth QAT senses + first-class orin variation

slug: `unsloth-qat-senses-first-class-orin-variation` · status: `exported` · from frame: `unsloth-qat-senses-first-class-orin-variation`

> senses serves unsloth/gemma-4-12B-it-qat-w4a16, live-validated on this Jetson AGX Orin as a first-class orin variation, while the thor/spark machine profiles stay intact so moving the setup to another architecture stays a profile pick, not a rework

## Tasks

### t1 — Catalog entry + per-model doc for unsloth/gemma-4-12B-it-qat-w4a16

- covers: c2, h2, c24, h21
- acceptance:
  - SupportedModel entry mirrors the coolthor gear (`role_hint`=multimodal, `tool_parser`=gemma4, quantization=compressed-tensors, doc=) with knobs cited from the downloaded config.json (262144 ctx, int4 pack-quantized, video+audio token ids); docs/gemma-4-12b-qat-w4a16.md exists with capability table marked pending-live-probe; uv run pytest tests/`test_catalog.py` tests/`test_catalog_tiers.py` green

### t2 — Orin card detection strategy (lobes/machines/orin.py)

- covers: c4, h3
- acceptance:
  - detect() resolves this box's facts (nvidia-smi name containing Orin / `compute_cap` 8.7 / device-tree Jetson AGX Orin) to orin; UNKNOWN cards still fall to base (existing detection tests untouched and green); one module + one register line per the `_registry` convention

### t3 — BOX: benchmark the incumbent coolthor on the current engine (BEFORE any swap)

- covers: c25, h22
- acceptance:
  - metric table recorded on this box against the running deployment: TTFT, decode tok/s from usage.`completion_tokens` (never chunk counts), MTP acceptance from SpecDecoding logs, KV pool from boot log — short/medium/long request shapes; saved as the baseline section of the evidence transcript draft

### t4 — BOX: snapshot ~/.lobes before any mutation

- covers: c27, h24
- acceptance:
  - a dated copy of .env + docker-compose.yml + docker-compose.shape.yml + docker-compose.audio.yml exists outside ~/.lobes and its existence is verified (diff empty against live) before t9 runs; restore procedure written into the runbook section

### t5 — Parameterize the senses lane --speculative-config (MTP off-switch)

- covers: c6, h5
- acceptance:
  - `MULTIMODAL_SPECULATIVE_CONFIG` env var with the current google-assistant literal as default; unset renders byte-identical (template-defaults golden unchanged); an explicit off sentinel omits the flag entirely (new test); `test_catalog.py` byte-guard updated; the off value is the recorded drop-MTP-if-unsupported path

### t6 — Builtin orin.toml profile (hypothesis knobs, measured-pending)

- depends on: t2
- covers: c4
- acceptance:
  - senses model=unsloth/gemma-4-12B-it-qat-w4a16, `TRITON_ATTN`, compressed-tensors, `gpu_mem_util`=0.45 hypothesis, `max_model_len`=262144 target — comments mark both MEASURED-PENDING until the live boot backfills; cortex/muse/worker feasible=false with the W4A4-needs-Blackwell rationale; `test_profile_schema` gains orin asserts; spark/thor asserts byte-untouched

### t7 — Builtin orin-lobe shape + iowait persistence + goldens regen

- depends on: t5, t6
- covers: c7, h6, c5, h4, c26, h23
- acceptance:
  - orin-lobe.toml hosts senses+embedder+reranker only (no stt/tts); overrides.senses carries the orin profile values so nothing clobbers; the Tegra iowait threshold renders persistently (exact knob name verified against lobes/gateway/`_pressure_policy.py`); goldens regenerated via regen.py; diff shows ONLY additions — zero byte changes to existing base/spark/thor goldens

### t8 — csv-mode GPU access knob (runtime:nvidia survives re-render)

- depends on: t5, t7
- covers: c10, h8
- acceptance:
  - a template knob or machine-strategy overlay emits runtime:nvidia GPU stanzas for csv-mode boards (orin opts in); default render for non-csv cards byte-identical per goldens; docs/machine-profiles.md documents the knob

### t9 — BOX: re-render from the branch checkout + boot + measure the budget

- depends on: t1, t3, t4, t5, t6, t7, t8
- covers: c3, h12, c9, c20, h16
- acceptance:
  - uv run lobes init --profile orin --shape orin-lobe --apply renders from the branch (detection resolves orin unforced); boot records BOTH accepted and refused `gpu_mem_util` values and the KV pool at the measured `max_model_len` (262144 target); the MTP draft is attempted and dropped via the off sentinel if vLLM refuses it, with the refusal captured; the int4-pack-quantized path booting on `sm_87` is the c3 proof; before-state divergences found live amend the frame, not get steamrolled

### t10 — BOX: proxy wiring — worker to Thor (new) + cortex to Spark (preserved)

- depends on: t9
- covers: c29, h26, c13, h14
- acceptance:
  - `WORKER_PEER_ORIGIN` (thor gateway :8000) + `WORKER_PEER_PROXY`=true + `WORKER_PEER_API_KEY` (thor's inbound key, operator-provided) set in .env; model=worker AND model=cortex through this gateway answer 200 with X-Lobes-Proxied-By naming thor/spark respectively; the pre-existing `PRIMARY_PEER_`\* lines survived the re-render byte-identical

### t11 — BOX: capability probe matrix + pressure check on the live checkpoint

- depends on: t9
- covers: c15, h10, c12, h9
- acceptance:
  - image: colour probe WITH opposite-colour negative control; video: same clip forwards vs reversed, directional answer required; audio: silent-drop-detecting probe (token-count delta + content assertion); reasoning: thinking request with sorted(message.keys()) dumped and `completion_tokens` reconciled against visible fields before any verdict (h20 discipline); tool-call smoke via the gemma4 parser; senses answers with NO 429 shed at the persisted iowait threshold while /proc/stat still shows inflated iowait; every verdict recorded pass OR fail

### t12 — Evidence transcript + docs + measured-value backfill

- depends on: t9, t10, t11
- covers: c9, h7, c23, h19, c28, h25, c22, h18, c21, h17
- acceptance:
  - docs/evidence/2026-08-04-accept-senses-unsloth-orin.txt committed with the incumbent baseline, boot measurements (accepted AND refused), per-capability verdicts incl. failures, proxy validations — keys redacted per the 2026-07-16 placeholder convention; orin.toml + orin-lobe.toml backfilled with MEASURED values and goldens re-regenerated; docs/orin-profiles.md updated to the new checkpoint; the per-model doc's capability table carries live verdicts, not card claims

### t13 — PR: version bump, full suite, untouched-proof, consumer coordination + follow-up issue

- depends on: t12
- covers: c1, h1, c11, h13, c19, h15
- acceptance:
  - version bumped per the every-PR rule; uv run pytest -n auto green; git diff proves zero byte changes to spark.toml/thor.toml/their goldens (orin-small untouched too); a follow-up issue files the role-name consumer migration (q3 decision) and the swap window is coordinated with the raw-id consumers; the announcement holds end-to-end: lobes status shows senses serving the unsloth id via the orin variation with the evidence committed

## Risks

- [unknown_nonblocking] KV arithmetic at the doubled window: the incumbent's 18.86 GiB pool held ~6x concurrency at 131072 but only ~3x at 262144 per request — util 0.45 at the full 262144 may be refused outright or the measured outcome may be a trimmed context; the plan treats whatever boots as the answer (task t9)
- [unknown_nonblocking] the swap window is mesh-visible downtime: the Spark forwards model=senses to this box, so its callers 404/timeout during the reboot; sequence the downtime and boot lobes sequentially per the Orin boot-ordering caveat (task t9)
- [unknown_nonblocking] worker proxy needs the Thor's inbound `GATEWAY_API_KEY`, an operator-provided secret not derivable by the plan; t10 blocks until the user supplies it (task t10)
- [unknown_nonblocking] video and audio verdicts may be NO (vLLM path gaps #101-style) — the capability goal is then partially unmet and reported honestly; the swap still ships (task t11)
