# Delivery Summary — orin decision models

plan: `orin-decision-models` · run: `partial` (every build and live task delivered; t14 — review and merge of PR #305 — still open) · date: `2026-10-14`
baseline: `devague summary skeleton`

## Intent

> The Jetson AGX Orin serves the mesh's decision models — Cloudflare Clef 27B (schema-scored decisions, Ampere-fit 4-bit) and, later, a fine-tuned Strands Decider 2B — through lobes, addressable from every mesh member

After: Any mesh member answers model=decider on POST /v1/systemone: the Orin serves Clef 27B (4-bit, co-resident with orin-embed) and any box may host a Strands Decider 2B; GET /capabilities advertises the decider role with its checkpoint, quant and feasibility, and a box that hosts none forwards to a verified member.

The operator's two questions were which 4-bit Clef suits the Orin's Ampere GPU and which quant the Strands 2B needs. The answers are: our own INT4 GPTQ W4A16 (group 64) build of Clef for the Orin, our own NVFP4 build for Blackwell, and bf16 for the Strands 2B.

## Planned Work

Quoted verbatim from the `devague summary` skeleton:

- `t1` — Spike: load a 4-bit Clef + its joint schema head under transformers on the Orin, beside the live orin-embed shape
- `t2` — Add the decider role to lobes/roles.py: ROLES and every per-role table
- `t3` — Add catalog entries for the two decision checkpoints, with a new decide task and a sidecar engine value
- `t4` — Ship the decider sidecar: a container that serves one decision checkpoint on POST /v1/systemone plus GET /health
- `t5` — Gateway: route POST /v1/systemone by the X-Lobes-Model header to the decider domain, body byte-for-byte, never shed
- `t6` — Readiness: probe decider lanes on their own health path
- `t7` — Capabilities advert + mesh: announce decider domains as their own roster key, route by domain, refuse conflicts
- `t8` — Render decider lanes: profiles/shapes tables, compose template services (no host ports), up target
- `t9` — Orin shape orin-decider (orin-embed + decider:general) and the exclusive heavy-lobe group; regenerate goldens
- `t10` — Docs: decider role page, colleague-stack justification, rollback procedure, a lobes route worked example
- `t11` — Live Orin acceptance: back up ~/.lobes, render orin-decider, serve Clef as decider:general beside orin-embed, measure on a labeled probe set
- `t12` — Live second-box acceptance: serve the stock Strands Decider 2B bf16 on spark2 as a decider domain, reached cross-box from every member
- `t13` — Orin main path: serve our culture-dev W4A16 Clef through the clef-vllm sidecar backend on vLLM 0.31; A/B against the t1 transformers NF4 baseline
- `t14` — PR: version bump, CHANGELOG, scope checks, evidence + delivery summary linked, review
- `t15` — /validate-delivery: run every confirmed obligation's behavioral test agent-side and file evidence + behavioral deltas
- `t16` — /summarize-delivery: write the planned-vs-actual closure artifact
- `t17` — Interim (d4): serve Cloudflare's original bf16 Clef as decider:general on the Spark through the decider sidecar
- `t18` — Build a calibration set and a disjoint labelled probe set of Clef-format decision records; publish privately to culture-dev
- `t19` — Quantize the Orin build: W4A16 INT4 GPTQ from Cloudflare/clef@2f3de3dd85f379784083b0814d997ab627200f0c with llm-compressor; upload privately to culture-dev
- `t20` — Quantize the Blackwell build: NVFP4 (W4A4) from Cloudflare/clef@2f3de3dd85f379784083b0814d997ab627200f0c with llm-compressor; upload privately to culture-dev
- `t21` — Decider sidecar: a clef-vllm backend (vLLM 0.31 pooling runner in-process + Cloudflare's joint head) and its image variant
- `t22` — Catalog: entries for Cloudflare/clef bf16 (interim) and our two culture-dev builds; retire the third-party Clef quant entries to candidate status
- `t23` — Verify both builds against the bf16 interim on the labelled probe set

## Actual Delivery

All work is on branch `scope/orin-decision-models` (draft PR #305, lobes-cli 0.86.0).

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | Spike GO (39db01a): a 4-bit Clef plus its joint head loaded under transformers on the Orin beside orin-embed (`docs/evidence/2026-10-14-spike-clef-nf4-transformers-orin.txt`). Its third-party NF4 checkpoint was later ruled out for serving by `d4`. |
| `t2` | delivered | The `decider` role in `lobes/roles.py` (99ee434), widened by `d1` to every role-enumerating surface (31f540d). |
| `t3` | delivered | Catalog entries with a `decide` task and the `decider-sidecar` engine (d62e66a); `lobes switch` refuses decide gears (865b313, `d1`). |
| `t4` | delivered | The decider sidecar, `clef` and `strands` backends, `Dockerfile.decider` with strands-decider pinned (e617dc6, ba21785, 98b01d8). |
| `t5` | delivered | `/v1/systemone` routed by `X-Lobes-Model`, body byte-for-byte, never shed (6f486d2, `d2`). |
| `t6` | delivered | Decider readiness on `/health`, never `/v1/models` (a3730f5). |
| `t7` | delivered | Decider domains as their own roster keys, per-domain forward, 409 on conflicting fingerprints (ef6d7c5). |
| `t8` | delivered | Per-domain `decider-<name>` services with no host port, `DECIDER_DOMAINS`, `lobes up decider` (0d32bac). |
| `t9` | delivered | The `orin-decider` shape and the Orin's exclusive heavy-lobe group (be6ae79). t11 then added a pinned 4 GiB KV pool to it (7e4f38d). |
| `t10` | delivered | `docs/decider.md`, the colleague-stack twelfth-role justification, the API row, CLAUDE.md, `lobes explain decider` (965e726). |
| `t11` | delivered | Live on the Orin (`docs/evidence/2026-10-14-accept-orin-decider.txt`, d79368b). The first four boots failed and needed two fixes: the hard-linked bundle (2c92ec3) and the KV pin (7e4f38d). Through the gateway: 418 questions, 0 failed, top-1 0.9354, agreement with bf16 0.9904, min host memory 13,036 MiB. The Thor and Spark gateways forward to it with `X-Lobes-Mesh-Member: orin`. |
| `t12` | delivered | The stock Strands 2B (216d1ccb, bf16) on spark2 as `decider:strands-general` (`docs/evidence/2026-10-14-accept-decider-2b-spark2.txt`, 05d41f1). Answered from the Orin, Thor and Spark gateways, each with `X-Lobes-Mesh-Member: spark2`. Needed one fix: the strands backend now honours `DECIDER_REVISION` (b58cf16). |
| `t13` | delivered | vLLM W4A16 adopted over the NF4 transformers path: agreement 0.988, 2.3-3.3x faster, serves past 8,192 tokens where NF4 refuses (240b6cf). |
| `t14` | partial | Version 0.86.0, CHANGELOG, Sonar fixes (bfbc8b5) and a draft PR #305 are done. Marking it ready, the review loop and the merge are still open. |
| `t15` | delivered | 14 evidence records (e1-e14, one per obligation, all `pass`) and deltas b4-b9 filed; all approved by the operator (2026-10-15), with lapses l1-l15. o11 passed on behaviour but not on its wording (delta b4). |
| `t16` | delivered | This file. |
| `t17` | partial | Stage A delivered: bf16 Clef served on the Spark (`docs/evidence/2026-10-14-accept-clef-bf16-interim-spark.txt`). Stage B dropped by `d6`; the interim was stopped after t11 and its override restored. |
| `t18` | delivered | Calibration set and disjoint labelled probe set, published privately to culture-dev (4a41e55; `docs/evidence/2026-10-14-build-clef-sets.txt`). |
| `t19` | delivered | `culture-dev/clef-27b-GPTQ-W4A16-g64` @4daaa224 (147390a and the quantize.py fixes). |
| `t20` | delivered | `culture-dev/clef-27b-NVFP4` @94093b9a, with every GatedDeltaNet projection kept bf16 (`d5`). |
| `t21` | delivered | The `clef-vllm` backend and `Dockerfile.decider-vllm` on vLLM 0.31.0 (97465af). |
| `t22` | delivered | The W4A16 build is the default decider; NVFP4 and the bf16 original are candidates; the third-party quant is demoted (675a4f7). |
| `t23` | delivered | `docs/evidence/2026-10-14-verify-clef-quants.txt` (aec3049): W4A16 on the Orin 0.9378 top-1 / 0.988 agreement; NVFP4 on the Spark 0.9330 / 0.9785; NVFP4 on the Thor gives flattened probabilities (cause not identified). |

## Mid-work Decisions

Quoted from the approved deviation records (`d3` was rejected and is superseded by `d4`):

- `d1` — t2 widens to make every role-enumerating surface decider-aware with minimal entries (cli/`_commands`/up.py `ROLE_SERVICE` decider -> decider service; gateway/`_config.py` decider as opt-in backend, infeasible by default; lobes measure `_FAMILY_BY_ROLE` 'decide' family; tests/`test_embed_lanes.py` ROLES pin; capabilities goldens regenerated); t3 additionally makes 'lobes switch' refuse task=decide gears (cli/`_commands`/switch.py) and keeps its regenerated tests/goldens/switch-plans.txt — Adding 'decider' to ROLES broke 16 tests outside roles.py; the plan put role tables in t2 and every consumer in later waves, which cannot keep the suite green per merge (TDD gate), and lobes measure + the embed-lanes ROLES pin were owned by no task. t3's catalog entries made lobes switch render a vLLM plan for decide gears (2B shown at 8192 vs its 4096 window), covered by no task.
- `d2` — t5 also owns the gateway service's `DECIDER_DOMAINS` environment passthrough line in lobes/templates/fleet/docker-compose.yml (t8 owns every other template edit and must not touch the gateway env block); t2's d1 merge likewise added `DECIDER_`\* gateway passthrough lines there — The doctor/gateway passthrough guard tests fail whenever gateway/`_config.py` reads an env key the compose gateway block does not pass through, so the reader and its passthrough line must land in the same merge; pinned in both briefs to keep same-wave t5/t8 file-disjoint
- `d4` — Clef serving is rebuilt on our OWN quantized checkpoints made from Cloudflare/clef@2f3de3dd (no third-party quant is served): a W4A16 build for the Jetson Orin (Ampere, Marlin) and an NVFP4 (W4A4) build for Blackwell (Spark/Thor), both uploaded to the culture-dev HF org (not jetson-ai-lab). Until the Orin build is served, decider:general runs Cloudflare's original bf16 weights with Cloudflare's own code in the decider sidecar on the Spark (interim 'healing'). A configured-but-down decider answers 503 + Retry-After. Supersedes proposed d3 (Aikimi NF4). — User decisions 2026-10-14 after the t1 spike and the third-party-quant safety audit: third-party quants (Aikimi NF4, JRamirez GPTQ) are not to be trusted for serving; one flavour per hardware class (Orin = W4A16, Blackwell = NVFP4); upload target culture-dev; bf16 on the Spark as the interim backend
- `d5` — t20's NVFP4 build keeps EVERY `linear_attn` (GatedDeltaNet) projection bf16, not the Orin build's narrower keep-list (`in_proj_a`/b only): llm-compressor 0.14.0's own Qwen3.5 NVFP4 recipe documents the fused GDN projections as incompatible with NVFP4. Also: t19/t20 run on the Spark BEFORE the t17 interim (user decision 2026-10-14: 'Spark first, then interim'), since quantization and the bf16 interim cannot co-reside. — upstream NVFP4 constraint on GatedDeltaNet; Spark memory (quant ~55 GB + interim ~56 GB > ~94 GB free)
- `d6` — Clef is kept ONLY as the W4A16 build on the Jetson AGX Orin: t17 stage B (gateway + mesh routing of the bf16 interim on the Spark) is dropped; decider:general is served from the Orin via the orin-decider shape (t9) and its live acceptance (t11). After t11 passes, the Spark bf16 interim (model-gear-decider-general, restore ~/.lobes override.yml from its backup) and the Thor NVFP4 sidecar are stopped. t17's evidence covers stage A only. — User decision 2026-10-14: 'When we're done with these, we can keep CLEF w4a16 only at Orin, and prepare for the fine tuning of Strandes Decider 2b', confirmed ('Confirmed'). W4A16 on the Orin was adopted in t13 (agreement 0.988).

Other decisions made during the run, not covered by a deviation:

- **Pin the decider's KV pool on the Orin (t11).** Left to vLLM, the pool is sized from the board's free memory, which moves with the neighbouring lanes. Two plain-run boots at the same settings got 94,890 and 131,072 tokens, and the rendered boots were OOM-killed at the 32 GiB cap. A new optional domain key `kv_cache_memory_bytes` now pins it, and orin-decider sets 4 GiB. That gives 56,661 tokens, 3.46x concurrency at 16,384.
- **Hard-link the clef-vllm bundle (t11).** `snapshot_download(local_dir=...)` made an 18 GB real copy inside the container, whose page cache starved the 32 GiB cgroup.
- **Honour `DECIDER_REVISION` in the strands backend (t12).** strands-decider's loader always fetched the repo HEAD.
- **Apply the Orin render by hand, keeping three live values (t11).** The Orin has no lobes CLI. The kept values are `GEMMA2_EMBED_MEM_LIMIT=8g`, the newer live `docker-compose.embed.yml`, and the dev version pin.
- **Domain name `strands-general` for the stock Strands checkpoint (t12),** per the task instruction to pick a neutral name.
- **Add `DECIDER_FEASIBLE=false` to the Thor and Spark gateways.** This turned out to be unneeded for routing (delta b8, lapse l14).

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|------------------------|-----------------|
| `t2` (`d1`) | Adding 'decider' to ROLES broke 16 tests outside roles.py; the plan put role tables in t2 and every consumer in later waves, which cannot keep the suite green per merge (TDD gate), and lobes measure + the embed-lanes ROLES pin were owned by no task. t3's catalog entries made lobes switch render a vLLM plan for decide gears (2B shown at 8192 vs its 4096 window), covered by no task. | `acceptable` |
| `t5` (`d2`) | The doctor/gateway passthrough guard tests fail whenever gateway/`_config.py` reads an env key the compose gateway block does not pass through, so the reader and its passthrough line must land in the same merge; pinned in both briefs to keep same-wave t5/t8 file-disjoint | `acceptable` |
| `t1` (`d4`) | User decisions 2026-10-14 after the t1 spike and the third-party-quant safety audit: third-party quants (Aikimi NF4, JRamirez GPTQ) are not to be trusted for serving; one flavour per hardware class (Orin = W4A16, Blackwell = NVFP4); upload target culture-dev; bf16 on the Spark as the interim backend | `needs-follow-up` |
| `t20` (`d5`) | upstream NVFP4 constraint on GatedDeltaNet; Spark memory (quant ~55 GB + interim ~56 GB > ~94 GB free) | `acceptable` |
| `t17` (`d6`) | User decision 2026-10-14: 'When we're done with these, we can keep CLEF w4a16 only at Orin, and prepare for the fine tuning of Strandes Decider 2b', confirmed ('Confirmed'). W4A16 on the Orin was adopted in t13 (agreement 0.988). | `acceptable` |
| `t9` / `t11` (no deviation) | The shape as first merged did not boot from its own render: the bundle copy and the unpinned KV pool each OOM-killed the engine at the 32 GiB cap. Fixed inside t11's scope (2c92ec3, 7e4f38d). Measured headroom under the cap is about 3.4 GiB of unreclaimable memory. | `risky` |
| `o11` (no deviation) | The obligation says machine-as-brain renders `DECIDER_FEASIBLE=false`; it renders no decider key at all. The absent key still reads infeasible, so the local answer is the same 404 (delta b4). | `acceptable` |
| `t10` / `t11` (no deviation) | The Orin's `/capabilities` reports the decider domain's fingerprint `max_model_len` as 262144, the catalog's native window, while the lane serves 16,384. | `needs-follow-up` |
| `t14` (no deviation) | The PR is still a draft; reviews, ready-for-review and merge are open. | `needs-follow-up` |

`d4`'s `needs-follow-up` stands for the NVFP4 build on the Jetson AGX Thor: it gives flattened probabilities there, cause not identified. Under `d6` it is not served.

## Evidence

- tests: `uv run pytest -n auto` at 05d41f1 — 6,738 passed, 17 skipped, 1 failed. The failure is `tests/test_gateway_pool_saturation.py::test_the_engine_plumbing_is_gone_from_the_server_module`; it also fails on `origin/main` (checked 2026-10-14 in a clean worktree) and passes in CI.
- behavioral: 20 tests (29 cases) backing obligations o1-o11, run at b58cf16 on 2026-10-14 21:33 UTC — all pass (evidence e1-e11).
- lint: `uv run black --check lobes tests`, `uv run isort --check-only lobes tests`, `uv run flake8 lobes tests` — clean.
- live transcripts: `docs/evidence/2026-10-14-spike-clef-nf4-transformers-orin.txt`, `...-build-clef-sets.txt`, `...-verify-clef-quants.txt`, `...-ab-clef-vllm-vs-sidecar-orin.txt`, `...-accept-clef-bf16-interim-spark.txt`, `...-accept-orin-decider.txt`, `...-accept-decider-2b-spark2.txt`.
- commits: 27d46cb..05d41f1 on `scope/orin-decision-models` (58 commits, with one merge of `origin/main`).
- PRs / issues: #305 (draft).

## Delivery Claims

Evidence e1-e14, deltas b1-b9 and lapses l1-l15 were approved by the operator on 2026-10-15. Confidence is capped where a lapse or a stated limit applies.

| Claim | Confidence | Evidence |
|-------|------------|----------|
| `c3` — Clef is served through a decision endpoint (`POST /v1/systemone`), not the chat surface; with no decider anywhere it answers 404 `role_infeasible`. | medium | e2: `tests/test_gateway_decider.py::test_no_decider_wired_is_role_infeasible`, `::test_decider_declared_infeasible_is_role_infeasible_even_when_wired` (2026-10-14). No test asserts a chat lane is never called. |
| `c9` — Clef takes the Orin's single heavy-lobe slot (exclusive with cortex/senses/associate). | medium | e10: `tests/test_orin_decider_shape.py::test_a_shape_hosting_associate_and_the_decider_is_refused` and the group pin; the CLI refusal path is tested with another group only. |
| `c15` — decider is opt-in: machine-as-brain never hosts it and an unwired decider 404s `role_infeasible`. | high | e11: `tests/test_decider_render.py::test_machine_as_brain_writes_no_decider_override`, `::test_no_builtin_shape_renders_any_decider_key_or_service`; wording drift recorded as b4. |
| `c25` — On the Orin, beside orin-embed, Clef 4-bit answers through the gateway with probabilities summing to 1, accuracy recorded, min host memory >= 2 GiB, latency recorded. | high | e12: `docs/evidence/2026-10-14-accept-orin-decider.txt` — 418 questions, 0 failed, max \|sum-1\| 0.0002, top-1 0.9354, agreement 0.9904, min MemAvailable 13,036 MiB, median 505 ms for inputs up to 1,024 tokens. |
| `c26` — A Strands 2B served bf16 on spark2 answers `decider:<domain>` from every other member with `X-Lobes-Mesh-Member`. | high | e14: `docs/evidence/2026-10-14-accept-decider-2b-spark2.txt` — Orin, Thor, Spark gateways each 200 with `X-Lobes-Mesh-Member: spark2` (one record each). |
| `c33` — decider contract: responsibilities end in `final_decision`; forbidden `repo_action`, `security_decision`. | high | e7: `tests/test_colleague_contract.py::test_decider_role_contract_is_pinned`. |
| `c34` — domains advertised and forwarded by name; conflicts refused; an unknown domain is 404 `model_not_found`. | high | e3, e4: `tests/test_gateway_decider.py::test_undeclared_domain_is_model_not_found_and_never_falls_back_to_general`, `tests/test_gateway_decider_mesh.py::test_two_members_one_domain_different_fingerprints_is_refused`; live, the Orin's `/capabilities` lists its own and a forwarded domain. |
| `c36` — decider readiness uses `/health`, not `/v1/models`. | high | e8: `tests/test_gateway_readiness.py::test_decider_ready_via_health_when_v1_models_404s`; live, `ready: true` on the Orin. |
| `c37` — decider containers publish no host port; the gateway is the only way in. | high | e9: `tests/test_decider_render.py::test_each_domain_renders_a_port_less_decider_service`; live, no published port on the Orin or spark2 and the plain-run port refused from the Spark. |
| `c38` — back up `~/.lobes` before the Orin change; rollback is restoring it. | medium | e13: the backup path is recorded; a rollback by restore was not exercised. |
| `c40` — decider domains are their own mesh roster key; conflicts refused. | high | e4: `tests/test_gateway_decider_mesh.py::test_member_c_forwards_each_domain_to_its_member` and the conflict test; live single-member forwards in both directions (Orin and spark2). |
| `c41` — domain selection rides in the `X-Lobes-Model` header; the body is forwarded byte-for-byte. | high | e1: `tests/test_gateway_decider.py::test_body_reaches_the_backend_byte_identical_and_the_answer_comes_back_unchanged`, `::test_no_header_resolves_to_decider_general`. |
| `c43` — decider is never shed under pressure. | medium | e5: `tests/test_gateway_decider.py::test_decider_is_never_shed_under_swap_and_iowait_pressure`, sampled at swap 95 / iowait 80, not at the 75/50 thresholds. |
| `c45` — each backend's own response shape passes through unchanged. | high | e6: `tests/test_decider_sidecar.py::test_body_passthrough_byte_identical_both_ways`; live, Clef's and strands-decider's different shapes both reached callers. |

Lapse ledger evidence:

| Lapse | Code | What |
|-------|------|------|
| `l1` | `provenance-missing` | Two challenge-pass links went wrong: scope entry s18 (mesh distributed-state lens) was recorded with no seed because its claim capture failed in the same batch (recaptured afterwards as c40); and s16 seeded q2 (the auto-generated 'contradiction with c19?' question) instead of q1, the wire-format question it actually grounds |
| `l2` | `assumption-for-measurement` | Strands decider's HTTP wire was read from the README only (a single noul example); the server source was not read, so the request/response shape for choice/score and the one-checkpoint-per-process reading of serve are README-derived, not code-verified |
| `l3` | `control-absent` | t2 (d1 widening): up.py/`_config.py`/measure changes were made code-first, then failing pins fixed — not test-first; only the measure stub test is new and was written after the code |
| `l4` | `provenance-missing` | t3: Clef context/`native_max_model_len` 262144 set from the Qwen3.8-27B base by the task agent, not read off a card; the main agent's scope research did read Cloudflare/clef config.json `max_position_embeddings`=262144, which matches, but the catalog comment does not cite it |
| `l5` | `assumption-for-measurement` | t4: the Clef backend's `load_release_model`/systemone call signature comes from the model card, not `joint_schema_model.py` source; unexercised until the t1/t11 live runs |
| `l6` | `control-absent` | t8: implementation written before tests/`test_decider_render.py` (red-by-import-error confirmed after the fact); and the generated decider-\* override was only yaml-checked, never run through a real 'docker compose config' |
| `l7` | `provenance-missing` | t8 edited lobes/cli/`_commands`/init.py and lobes/runtime/`_compose.py`, outside its brief's file list (needed: override generation + Dockerfile scaffolding live there); not a separate approved deviation |
| `l8` | `control-absent` | t1: third-party code from Aikimi/clef-nf4 (`clef_runtime`/\*.py, `joint_schema_model.py`) was EXECUTED on the Orin (container root, GPU, default bridge network, rw mount of that model's cache dir) before it was audited; the audit was done afterwards, only when the user asked whether the repo is safe |
| `l9` | `control-absent` | t7 agent wrote tests after the implementation and never saw them fail first (TDD red step skipped) |
| `l10` | `assumption-for-measurement` | t21 clef-vllm backend assumes vLLM 0.31's pooling runner works on hybrid GDN+attention Qwen3.5 (vLLM's own comments say no hybrid pooling models exist), that a background-thread LLM() start works, that the v0.31.0 image ships uv, and that its arm64 build carries `sm_87` kernels — all unverified until t13 |
| `l11` | `assumption-for-measurement` | status update at 10:06 UTC reported the W4A16 run at 48/65 with ~20 min left; it had already been stalled at 48/65 since 09:38 (no log-timestamp check) — the hang was found only at the next tick |
| `l12` | `assumption-for-measurement` | introduced a live 'docker update --cpus' thermal governor on a GPU container without checking its device-cgroup side effect; it caused cudaErrorNotPermitted twice and destroyed a finished 40-min GPTQ result |
| `l13` | `assumption-for-measurement` | l12 and my status message claimed 'docker update' CAUSED the cudaErrorNotPermitted exits; only correlation was observed (2 exits 5 s / 27 s after an update; a 0.8B repro with 9 toggles did not fail). The operator attributes the faults to overheating with the evidence lost. Cause unproven. |
| `l14` | `assumption-for-measurement` | t11/t12: added `DECIDER_FEASIBLE`=false (+ passthrough) to the live Thor and Spark gateways on the strength of a memory note about OTHER opt-in roles (absent key skips the mesh), without first reading the decider's own /v1/systemone path; it turned out not to be needed (delta b8), and b8's 'not needed' is itself read from code, not tested with the key absent |
| `l15` | `control-absent` | t11: two background watchers used `pgrep -f <pattern>` inside ssh, which matches its own remote shell, so they could never fire (one ran ~35 min); a third used 'docker logs --since HH:MM:SS', an invalid format, so it could not detect an engine failure — status at ~20:36 UTC leaned on that broken watcher until it was replaced |

How the approved lapses cap the claims above: `l4` (Clef's 262144 context taken from the base model) is the likely source of the advert gap in the Drift table. `l14` touches c26/c40 only through delta b8, which rests on reading the code, not on a test. `l2` and `l5` (wire formats read from docs) were retired by the live runs in t11 and t12, which exercised both backends end to end.

## Remaining Work / Follow-up

- **PR #305 (t14):** mark ready, wait for the Qodo, Copilot and Sonar reviews, fix or answer every thread, then merge — the operator.
- **Advert gap:** the decider domain's fingerprint `max_model_len` should report the served window (16,384 on the Orin), not the catalog's 262144 — follow-up issue.
- **Thin margin on the Orin:** the decider container runs at its 32 GiB cap with about 3.4 GiB of unreclaimable headroom. A larger KV pin or a second engine there needs a higher `mem_limit` — keep in mind before changing the shape.
- **NVFP4 on the Thor:** flattened probabilities, cause not identified. Not served (`d6`); only matters if a Blackwell Clef is wanted again.
- **Strands 2B fine-tune (outside this plan):** student `strands-decider-2B-qwen3.5-v1-2610`, teacher Clef W4A16 on the Orin. The stock baseline on the same probe set is top-1 0.756 against Clef's 0.9354, weakest on the `incident` source at 0.655. It silently truncates inputs past its 4,096-token window. The work needs jev-factory retargeted from v21 to 2610, plus an nvsh `tool_jev` adapter for `/v1/systemone`.
- **Calibration of decider probabilities in lobes:** confirmed by the operator as new scope, not this plan — offer to file an issue.
- **Spare containers:** the Orin's plain-run W4A16 container is stopped and kept as a fallback; the Thor's NVFP4 container is stopped. Remove both once the shape has run for a while.
