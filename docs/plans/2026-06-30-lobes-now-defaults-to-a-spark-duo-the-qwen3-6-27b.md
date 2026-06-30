# Build Plan — lobes now defaults to a Spark duo: the Qwen3.6-27B-MTP primary paired with a Gemma 4 12B NVFP4 multimodal worker that takes over the normal/middle tier, replacing the text-only Qwen3-14B (demoted to a legacy candidate profile); the duo gives Spark multimodal senses, a diverse non-Qwen mind, and a DSpark speculative-decoding test target, with old models kept behind explicit legacy profiles.

slug: `lobes-now-defaults-to-a-spark-duo-the-qwen3-6-27b` · status: `exported` · from frame: `lobes-now-defaults-to-a-spark-duo-the-qwen3-6-27b`

> lobes now defaults to a Spark duo: the Qwen3.6-27B-MTP primary paired with a Gemma 4 12B NVFP4 multimodal worker that takes over the normal/middle tier, replacing the text-only Qwen3-14B (demoted to a legacy candidate profile); the duo gives Spark multimodal senses, a diverse non-Qwen mind, and a DSpark speculative-decoding test target, with old models kept behind explicit legacy profiles.

## Tasks

### t1 — Parser: add a runtime/_parser.py infer_parser rule for the Gemma 4 12B id

- covers: h6
- acceptance:
  - infer_parser(<gemma-id>) returns the tool-call-parser the served Gemma build actually uses (verified against vLLM); a unit test asserts it and the tests/test_catalog.py tool_parser==infer_parser assertion passes for the Gemma entry

### t2 — Catalog: add the Gemma multimodal gear, reframe tiers to main/minor/multimodal, demote the 14B

- depends on: t1
- covers: c14, c16, h6, h13
- acceptance:
  - catalog.py defines a SupportedModel role_hint=multimodal for the Gemma 4 12B NVFP4 id (task=generate, native-MTP speculative_config, doc filename); nvidia/Qwen3-14B-NVFP4 role_hint becomes candidate; TIER_ROLE maps main->primary, minor->minor, multimodal->Gemma with normal->multimodal and hard->main and cheap->minor back-compat; test_catalog passes

### t3 — Compose: default-on vllm-multimodal (vision+audio+native-MTP), 14B behind a legacy profile, DSpark toggle off

- depends on: t2
- covers: c8, c17, c15, h15
- acceptance:
  - templates/fleet/docker-compose.yml has a DEFAULT (non-profile) vllm-multimodal service serving the Gemma NVFP4 build WITHOUT --language-model-only (vision+audio active) with a native-MTP --speculative-config; the 14B vllm-middle moves behind a legacy compose profile; a DSpark draft env toggle defaults OFF; env.example carries MULTIMODAL_* defaults + retuned GPU utils; a template-lint test asserts the default service set and that catalog mtp_compose_command_items match the template

### t4 — Gateway: route model=main/minor/multimodal with normal/hard/cheap back-compat aliases

- depends on: t2
- covers: c13, h10, c2
- acceptance:
  - the gateway resolves model=main->primary, model=minor->minor, model=multimodal->Gemma; model=normal->multimodal, model=hard->main, model=cheap->minor as back-compat; absent-tier upward fallback preserved; unit tests cover every alias and a fallback case

### t5 — Serve: lobes serve brings up the main+multimodal duo by default, legacy gears behind explicit flags

- depends on: t3
- covers: c15, c13, h7, h9
- acceptance:
  - lobes serve (dry-run plan and --apply) composes up exactly the main + multimodal generate gears with no extra flags and NO legacy gear; minor/14B require an explicit profile/flag; a test asserts the serve plan = {main, multimodal} and excludes minor/middle

### t6 — Pressure: redefine the degraded-mode downgrade target under main/minor/multimodal + lobes status --pressure

- depends on: t4
- covers: h7
- acceptance:
  - the degraded-mode pressure policy defines a valid downgrade target under main/minor/multimodal (proposed: under pressure, main/multimodal generate traffic degrades to minor; documented that multimodal is a capability, not a cheaper rung); lobes status --pressure reports the tier ceiling in the new vocabulary; a unit test asserts a degraded sample downgrades to minor

### t7 — Live validation: load-test the Gemma gear on the DGX Spark (boot, vision, audio, MTP, budget)

- depends on: t2, t3
- covers: c8, c17, c12, h1, h5, h8, h16
- acceptance:
  - on the DGX Spark under the production compose, the chosen Gemma NVFP4 build boots with vision+audio + native MTP (measured draft acceptance >0), answers a model=multimodal image request and a model=multimodal audio request correctly, and the fleet (main+multimodal+embed+rerank) measures <1.0 total GPU util; numbers recorded; catalog status promoted configured->load-tested

### t8 — Smoke test: assert the duo is reachable and a legacy profile is still selectable

- depends on: t5, t4
- covers: c7, h14
- acceptance:
  - a committed smoke test asserts model=main returns text, model=multimodal returns valid output for an image round-trip AND an audio round-trip, and an explicit legacy profile (14B) boots and serves; the test runs in CI (live-gated or mocked transport)

### t9 — Docs: per-model Gemma doc, gateway-fleet topology+pressure, 14B demotion, duo-default in README/CLAUDE

- depends on: t2, t6
- covers: c1, c4, c5, c6, h11, h12
- acceptance:
  - docs/gemma-4-12b-nvfp4.md documents the multimodal gear (vision+audio, native MTP, DSpark experiment, tier alias, measured budget); gateway-fleet.md describes the main/minor/multimodal topology + the redefined pressure seam; qwen3-14b-nvfp4.md notes the candidate/legacy demotion; CLAUDE.md/README reflect the duo default; markdownlint passes

## Risks

- [unknown_nonblocking] Exact HF checkpoint pick for the default Gemma 4 12B NVFP4 gear. NVFP4 12B builds provably exist; leading candidate sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4 (same publisher as the primary, NVFP4+MTP); alts AxionML/Gemma-4-12B-NVFP4, coolthor/gemma-4-12B-it-NVFP4A16. Verify+pick during t7. (task t7)
- [unknown_nonblocking] Gemma 4 --tool-call-parser is unconfirmed (Gemma has no strong native tool format). Determine it and add the matching infer_parser rule so the catalog test stays green. (task t1)
- [unknown_nonblocking] Whether the chosen Gemma build loads non-gibberish on the nv26.04 Blackwell vLLM image (lower risk: Gemma4UnifiedForConditionalGeneration is registered, auto-detects NVFP4, not the Qwen3.5 FLA arch). Verify before promoting to load-tested. (task t7)
- [unknown_nonblocking] Exact --speculative-config method+JSON for Gemma 4 native MTP (default) and the DSpark draft (deepseek-ai/dspark_gemma4_12b_block7). Confirm the method string against the served checkpoint config. (task t3)
- [unknown_nonblocking] Measured GPU util for the multimodal Gemma gear (vision+audio embedders + image/audio KV vs the 14B 0.12) and whether the duo needs primary/util retuning to fit 128GB. (task t7)
- [unknown_nonblocking] Exact degraded-mode downgrade target under main/minor/multimodal is a design decision (proposed: degrade to minor) pending user confirmation in t6. (task t6)
