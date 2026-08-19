# Build Plan — qwen3.8 cortex upgrade

slug: `qwen3-8-cortex-upgrade` · status: `exported` · from frame: `qwen3-8-cortex-upgrade`

> lobes upgrades the cortex checkpoint from unsloth/Qwen3.6-27B-NVFP4 to unsloth/Qwen3.8-27B-NVFP4 on the DGX Spark, served on a vLLM engine new enough to register the Qwen3.8 arch, with measured GB10 budgets and the option of 1M YaRN context

## Tasks

### t1 — Baseline: benchmark the INCUMBENT 3.6 cortex on the current engine, live on the Spark, before anything changes

- covers: c9, h6, c18, h13
- acceptance:
  - transcript records TTFT, decode tok/s at three shapes (short/medium/512+ gen) measured via usage.`completion_tokens`, MTP acceptance from SpecDecoding logs, and KV pool — with unique prefixes and fresh images
  - the live before-state (digest, 0.44@262144, served id) is re-checked on the box and recorded, not assumed
  - the transcript is committed under docs/evidence/ BEFORE any swap commit lands

### t2 — Engine spike: newest OFFICIAL vllm/vllm-openai nightly digest boots 3.8 NVFP4 standalone on the GB10 at 262144

- depends on: t1
- covers: c13, h9
- acceptance:
  - the candidate digest's resolved vLLM version is recorded; a standalone container serves unsloth/Qwen3.8-27B-NVFP4 with MTP armed and answers a completion
  - if the official image fails, the exact failure is recorded and the spark-arena fallback is tried and digest-pinned — official-first order is documented

### t3 — Catalog: add the config-verified 3.8 entry, demote 3.6 to candidate (lobes/catalog.py + catalog tests only)

- covers: c2, h2
- acceptance:
  - every field of the new entry cites the checkpoint's published config files fetched at implementation time; tokenizer.json truncation re-verified null
  - 3.6 gets `role_hint` candidate, stays selectable via lobes switch; uv run pytest tests/`test_catalog.py` tests/`test_catalog_tiers.py` passes

### t4 — Parser: `_RULES` gains qwen3.8 markers -> `qwen3_coder` (lobes/runtime/`_parser.py` + tests/`test_parser.py` only)

- covers: c6, h4
- acceptance:
  - `infer_parser`('unsloth/Qwen3.8-27B-NVFP4') == '`qwen3_coder`'; the catalog test asserting `tool_parser`==`infer_parser`(id) passes

### t5 — Templates + profiles: bump the shared digest for ALL SIX lanes together, flip the primary id, wire the 1M YaRN knobs (templates/\*, profiles spark.toml + spark-lobe.toml)

- depends on: t2
- covers: c4, h11, c29, h21, c13
- acceptance:
  - one digest-form pin (from t2) replaces the old default on all six `VLLM_NIGHTLY_IMAGE` lanes in one commit — never a subset; the compose comment records the resolved vLLM version
  - grep finds no mutable tag in any committed compose/env line for the cortex lane
  - spark-lobe renders `max_model_len`=1048576 with the YaRN hf-overrides on `text_config`.`rope_parameters` (all mrope fields preserved), `VLLM_ALLOW_LONG_MAX_MODEL_LEN`=1, and util marked as a hypothesis pending the t7 boot

### t6 — Id sweep: every remaining surface naming the old id updates (roles.py, gateway/`_config.py`, whoami.py, explain/catalog.py, machines/orin.py, remaining tests) — no catalog/parser/template files

- depends on: t3, t5
- covers: c8, h5
- acceptance:
  - grep for unsloth/Qwen3.6-27B-NVFP4 finds it only in candidate/demotion/history contexts; full test suite passes

### t7 — Live swap + 1M boot on the Spark deployment per the playbook: boot 3.8 at 262144 first, then extend to 1M YaRN, reclaiming co-resident gear budget if needed

- depends on: t1, t2, t5
- covers: c1, h1, c5, h3
- acceptance:
  - a live GB10 boot serves 3.8 at `max_model_len`=1048576 and a gateway chat completion answers 200 for model=cortex and the raw id
  - every committed `gpu_mem_util`/`max_model_len` is a value the boot actually accepted; any gear trim/drop exercised under the c16 reclaim decision is recorded
  - no Thor/Orin deployment state is touched

### t8 — Acceptance gates on the live 1M lane: six-lane probes, long-prefill operability, strict tools, YaRN quality, benchmark vs baseline

- depends on: t7
- covers: c20, h15, c22, h10, c23, h18, c27, h19
- acceptance:
  - all SIX digest lanes pass their correctness probes, embed-deep's result recorded explicitly, not inferred
  - a >=200K-token prompt completes through the gateway with no layer timing out; the effective timeout values are recorded
  - a strict:true + `enable_thinking`=true tool call returns a schema-valid call on the new engine (plugin re-verified, ported, or superseded — never carried forward unverified)
  - short-context quality compared on the same prompts native-vs-1M-YaRN, and decode/TTFT/MTP-acceptance compared against the t1 incumbent baseline — numbers, not impressions

### t9 — Evidence + docs + release: transcript, per-model doc, rollback recipe, version bump

- depends on: t8
- covers: c11, h8, c19, h14, c21, h16, c28, h20
- acceptance:
  - the evidence transcript lands under docs/evidence/ IN the promoting PR; no surface says VALIDATED without it
  - docs/qwen3.8-27b-nvfp4.md exists; CLAUDE.md served-model paragraph updated; measured knobs folded back into profile comments
  - the playbook/PR carries an executable rollback section quoting the old digest and exact .env lines
  - CHANGELOG entry + version bump per the every-PR-bumps rule

### t10 — Rollout notes to consumers and peers: the served-id repoint, addressed to every raw-id pinner

- depends on: t7
- covers: c10, h7, c17, h12, c25, h22
- acceptance:
  - the notes name each consumer repo (culture/colleague, eidetic, reachy-mini-cli, the lobes agent's culture.yaml) and every peer .env mirror, with the new id and the date — none learns via a 404
  - the notes contain zero Thor/Orin deployment changes; Jetson boxes byte-identical before and after

## Risks

- [unknown_nonblocking] the in-tree `VLLM_NIGHTLY_IMAGE` default changes for future scaffolds on cards that never booted it (`sm_110`/Orin) — UNVALIDATED there per #108 until a later boot
- [unknown_nonblocking] MTP self-draft acceptance at 1M-YaRN depth is unmeasured anywhere; t8 measures it on this deployment for the first time (task t8)
- [unknown_nonblocking] one full-context request can monopolize the KV pool for a multi-minute prefill while shorter cortex requests queue — no per-request KV fairness in v1
