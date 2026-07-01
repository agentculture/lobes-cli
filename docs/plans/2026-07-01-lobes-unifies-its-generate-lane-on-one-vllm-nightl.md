# Build Plan — lobes unifies its generate lane on one vLLM nightly: the Qwen 3.6 27B primary migrates onto the nightly image the Gemma 4 12B gear already runs, both serve with MTP speculative decoding, and lobes publishes a same-engine head-to-head of the two gears decode/prefill throughput and MTP draft acceptance

slug: `lobes-unifies-its-generate-lane-on-one-vllm-nightl` · status: `exported` · from frame: `lobes-unifies-its-generate-lane-on-one-vllm-nightl`

> lobes unifies its generate lane on one vLLM nightly: the Qwen 3.6 27B primary migrates onto the nightly image the Gemma 4 12B gear already runs, both serve with MTP speculative decoding, and lobes publishes a same-engine head-to-head of the two gears decode/prefill throughput and MTP draft acceptance

## Tasks

### t1 — Before-state verification + baselines (docs)

- covers: c3, h6, h7
- acceptance:
  - docs/vllm-nightly-migration.md records, verified in-repo, that the primary/embed/rerank services pin nvcr.io/nvidia/vllm:26.04-py3 (vLLM 0.19.0) today and the gemma service pins the nightly image
  - the doc records the baselines to beat (27B ~19 tok/s decode + 72-79 percent MTP accept, Gemma ~23 tok/s no-spec) and cites the generate-lane (model=main/multimodal) mesh traffic that justifies the benchmark

### t2 — Spike: 27B primary serves on nightly (standalone, no fleet mutation)

- covers: h1
- acceptance:
  - a standalone 27B NVFP4-MTP container on the pinned nightly image returns /health 200 with Qwen3_5ForConditionalGeneration + Qwen3_5MTP resolved in logs, and the working --speculative-config method string (qwen3_5_mtp or its nightly replacement) plus the quant flag are recorded
  - the spike records a passing qwen3_coder tool-call round-trip, MTP draft acceptance > 0, and decode tok/s; a hard regression (gibberish / lost MTP / rejected quant) is recorded as a STOP with the pinned 0.19.0 rollback noted

### t3 — Spike: embed + rerank pooling gears serve on nightly (standalone)

- covers: h2
- acceptance:
  - a standalone embed container on the nightly image returns a correct-dimension vector from POST /v1/embeddings (--runner pooling --convert embed working)
  - a standalone rerank container on the nightly image returns ordered scores from /v1/rerank and /v1/score (--convert classify working); findings recorded per gear

### t4 — Migrate primary + embed + rerank default images to the nightly backend (compose/env)

- depends on: t2, t3
- covers: c12, c13
- acceptance:
  - the fleet compose + env.example templates pin the primary, embed, and rerank services to the nightly image at a single pinned digest (replacing nvcr.io/nvidia/vllm:26.04-py3), and set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 where the nightly cudagraph estimate requires it
  - the 27B --speculative-config method matches the t2 spike finding; a template test asserts every vLLM gear service uses the nightly image while the realtime Parakeet/Chatterbox sidecars are unchanged

### t5 — Wire the Gemma DSpark speculative_config catalog-driven + round-trip test

- depends on: t4
- covers: c14, h3
- acceptance:
  - lobes/catalog.py carries the gemma speculative_config for the DSpark draft_model route and mtp_compose_command_items emits the gemma --speculative-config items exactly as for the 27B primary
  - a round-trip test asserts the gemma --speculative-config items add/remove through lobes switch; the wiring is present and testable but NOT yet default-on (the default flip is gated on the t7 verdict)

### t6 — Measure Gemma DSpark MTP + assemble the same-nightly head-to-head

- depends on: t4, t5
- covers: c15, h4, h5
- acceptance:
  - on the serving gemma gear the DSpark draft loads with no vocab/tokenizer mismatch and yields > 0 percent draft acceptance; acceptance percent and decode tok/s vs the ~23 tok/s no-spec baseline are recorded
  - a committed head-to-head table records both generate gears decode + prefill tok/s + MTP accept under the SAME pinned nightly version, states the util/max-model-len/co-resident-vs-standalone method, and flags the 12B-vs-27B size difference

### t7 — Commit the Gemma MTP verdict (flip guards + default-on if it wins, else document negative)

- depends on: t6
- covers: c16
- acceptance:
  - if the t6 DSpark acceptance/speedup beats the no-spec baseline: the gemma catalog speculative_config + compose items are made default-on and the three guard tests (test_gemma_has_no_speculative_config, test_fleet_compose_multimodal_vision_active_no_spec_decode, MTP-items drift guard) are flipped to the spec-enabled invariant and green
  - otherwise the gemma gear keeps no spec-decode and docs/gemma4-mtp-draft.md records the measured negative; exactly one outcome is committed

### t8 — Migrate opt-in minor (4B) + legacy 14B gears to nightly (trailing)

- depends on: t5
- covers: c13
- acceptance:
  - the opt-in minor (4B bf16, COMPOSE_PROFILES=minor) and legacy 14B NVFP4 (COMPOSE_PROFILES=middle) services are pinned to the nightly image and each answers a generate probe when its profile is enabled
  - this task is trailing (may land after the default-on gears): if a gear cannot serve on nightly its residual is documented, not silently dropped

### t9 — After-state / shipped-state docs + success-artifact verification

- depends on: t6, t7
- covers: c1, c2, c5, c9, c10, c11, h8, h9, h10, h11, h12
- acceptance:
  - docs/gateway-fleet.md and the per-gear docs record the shipped state: every vLLM gear on the nightly image (docker ps / lobes status verifiable), the four success artifacts committed, and the scope line held (realtime sidecars + no draft-head-training untouched)
  - the docs frame the audience (operators + mesh addressing model=main/multimodal) and why-same-engine-matters without contradicting the committed benchmark numbers

## Risks

- [unknown_nonblocking] the 27B qwen3_5_mtp method is already deprecated on 0.19.0; nightly may require method=mtp or shift the Qwen3.5 hybrid/FLA attention path — the primary could lose MTP or gibber. Resolved by the t2 spike; fallback method=mtp, rollback the pinned 0.19.0 image (task t2)
- [unknown_nonblocking] nightly vLLM is memory-hungry (may need primary down); the whole fleet co-resident on nightly may exceed the GB10 budget, forcing sequential/standalone benchmark measurement (task t6)
- [unknown_nonblocking] the exact nightly image tag/digest to standardize the whole fleet on (audio-extra image vs a lighter text-only nightly at the same version) — a single pinned digest is the simplest same-engine guarantee (parked v1) (task t4)
- [unknown_nonblocking] DSpark recommended num_speculative_tokens on this serving checkpoint (config block_size=7; the disabled experiment used 3) — measured at serve time (parked v2) (task t6)
- [unknown_nonblocking] whether the opt-in minor/14B live-validation blocks this effort or trails as a follow-up (parked v3) — planned as trailing (t8 dep t5), residual documented if a gear cannot serve (task t8)
- [follow_up] native gemma4_assistant draft (google/gemma-4-12B-it-assistant) as the escalation route if DSpark is insufficient (parked v4, per #75) (task t7)

## Execution waves (from `devague plan waves` — scheduling metadata, not orchestration)

- **wave 0:** t1, t2, t3 — parallel; **file-disjoint** (t1 → new
  `docs/vllm-nightly-migration.md`, t2 → `docs/qwen3.6-27b-text-nvfp4-mtp.md`,
  t3 → `docs/qwen3-embedding-0.6b.md` + `docs/qwen3-reranker-0.6b.md`). All three
  are standalone spikes / doc-writes with **zero default-fleet mutation** — the
  de-risk pass that must clear before any image flip.
- **wave 1:** t4 — flip primary/embed/rerank default images (compose + env).
  Gated on the wave-0 spikes passing.
- **wave 2:** t5 — Gemma DSpark wiring (catalog + compose-items + tests).
- **wave 3:** t6, t8 — measure + head-to-head (t6) alongside the trailing
  minor/14B migration (t8). Formally parallel; note t6 is measurement/docs while
  t8 edits compose/catalog, so verify file-disjointness at fan-out.
- **wave 4:** t7 — commit the Gemma verdict (guards + default-on, or documented
  negative).
- **wave 5:** t9 — shipped-state docs + success-artifact verification.

> **Shared-file caveat for `/assign-to-workforce`:** t4, t5, t8 all edit the
> fleet `docker-compose.yml`; the dependency chain (t4 → t5 → t8) serializes
> them deliberately, so no two same-wave tasks write the same compose file. The
> dependency graph sequences *content*; file-disjointness within a wave is the
> operator's check at fan-out.
