# Delivery Summary — DSpark speculation on the Spark cortex

plan: `dspark-speculation-on-the-spark-cortex` · run: `complete` · date: `2026-08-24`
baseline: `devague summary skeleton`

## Intent

> The Spark's cortex got faster without swapping the model or the engine: a DSpark block-speculative drafter now rides the deployed vLLM NVFP4 lane, measured against its own 19.9-24.0 tok/s baseline on the same box — and the repo records SGLang as a known engine axis, so the framework that first demonstrated this is in the history rather than lost to a forum post.

After: The Spark's cortex serves the same unsloth/Qwen3.8-27B-NVFP4 checkpoint on the same vLLM engine, with a measured decode figure taken under a DSpark speculative config, and an evidence transcript that reports the result either way.

## Planned Work

- `t1` — Add `ENGINE_SGLANG` to the catalog's engine axis
- `t2` — Add the RadixArk NVFP4 target and DSpark drafter as revision-pinned candidate catalog entries
- `t3` — Write docs/dspark-speculation.md — the SGLang recipe and the lossless framing, both marked UNVALIDATED-by-this-repo
- `t4` — Build scripts/spike-preflight.sh — peer-state read, rendered-argv proof, stop announcement, restore verification
- `t5` — Build scripts/spec-arms.py — the three-arm, multi-shape measurement runner
- `t6` — Run the live three-arm spike on the DGX Spark and land the evidence transcript
- `t7` — Fold the measured result back into the docs and bump the version

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | `ENGINE_SGLANG` added to the engine axis; `ENGINES` pinned to a 3-value tuple; a test enumerates every current catalog id and asserts `serves_with_vllm()` is unchanged. commit `c5f4068` |
| `t2` | delivered | `RadixArk/Qwen3.8-27B-NVFP4` + `RadixArk/Qwen3.8-27B-DSpark` as `role_hint=candidate`, `engine=ENGINE_SGLANG`, via a new `hf_revision` field. Both pins verified HTTP 200 on the Hub. commit `b302576` |
| `t3` | delivered | `docs/dspark-speculation.md` with a 3-label provenance vocabulary (PUBLISHED-ELSEWHERE / MEASURED-HERE (dated) / STRUCTURAL). markdownlint 0 errors. commit `0d28df0` |
| `t4` | delivered | `scripts/spike-preflight.sh` + 22 offline tests. Its argv proof caught the deployment defect below. commit `654c9bc` |
| `t5` | delivered | `scripts/spec-arms.py` + 17 offline tests; `--combine` refuses to backfill a MISSING arm. commit `23b2a42` |
| `t6` | delivered | Live spike: 9 runs / 5 arms / 3 shapes / 3 windows / 2 targets. `docs/evidence/2026-08-24-spike-dspark-cortex-spark.txt` (286 lines) + 9 raw JSON transcripts. commits `1ef9e99`, `b4dcf80` |
| `t7` | delivered | Result folded into `docs/qwen3.8-27b-nvfp4.md` and `docs/dspark-speculation.md`; version `0.59.1` -> `0.60.0`. commit `dde9bf4` |

## Mid-work Decisions

- `d1` — Add a FOURTH measurement point: run the DSpark arm at 262144 — the drafter's own declared `max_position_embeddings` and the target's native ceiling — in addition to the announced 786432 trade-down. — Operator request during the live run. It is not merely a memory-headroom question: at 786432 the TARGET is YaRN-extended while the 1.36B drafter declares only 262144 positions, so the drafter is being asked to operate beyond its trained/declared range — a plausible acceptance-suppressing confound distinct from the FP8-vs-W4A4 mismatch already recorded as c20/r1. At 262144 no YaRN extrapolation is asked of either model, KV demand drops far below the 40.76 GiB ceiling that refused the first boot, and DSpark is measured in the regime it was actually built for. Cost: one additional lane restart inside the already-open cortex-down window.
- `d2` — Add a FIFTH arm on a DIFFERENT TARGET checkpoint: huginnfork/Qwen3.8-27B-NVFP4A16 (NVFP4 weights, 16-BIT activations, compressed-tensors, quantized from Qwen/Qwen3.8-27B, sha 6916a5bb185e57c6e32bcffdc13a92fdea3b4095), served with the same DSpark drafter. — Operator request, and it is the sharpest available test of the c20/r1 hypothesis rather than a second opinion on it. vLLM's DSpark speculator consumes the target's MEAN-POOLED AUX HIDDEN STATES — i.e. ACTIVATIONS. The drafter was SpecForge-trained against a Qwen3.8-27B-FP8 target; the deployed cortex is W4A4, which quantizes activations to FP4. An A16 target keeps activations at 16-bit, so its hidden states should sit numerically far closer to the FP8 target the drafter learned from. If the prose acceptance collapse (28.6%) is caused by activation quantization, THIS is the arm that recovers it. Classified RISKY because it violates frame boundary c12 (the deployed checkpoint does not change): it requires serving a different target checkpoint, a ~16 GiB download, and it is no longer a single-variable test against the incumbent.
- `d3` — Complete the 262144 (drafter-aligned) window with its MISSING arms: mtp-n2 and none at 262144, so that window carries a full three-arm comparison instead of a lone dspark data point. — Operator request. Deviation d1 measured dspark at 262144 but the other two arms were only measured at 786432, so the drafter-aligned window currently has no floor and no incumbent to compare against — exactly the interpretability gap that arm 3 fixed at 768K. c27's own rationale (arm 3 is what makes arms 1 and 2 interpretable) applies per-window, not once globally.
- `d4` — ADOPT the DSpark arm as the deployed cortex configuration: unsloth/Qwen3.8-27B-NVFP4 + RadixArk DSpark drafter at `max_model_len`=262144, replacing self-hosted MTP n=2 at 1048576. — Operator decision after reading the five-arm result. t6 was scoped as a SPIKE that ends by restoring the incumbent (c3/h7); adopting a measured arm is a change of deployed serving configuration and therefore beyond that scope. Two consequences are accepted deliberately: (1) the advertised 1M YaRN window is withdrawn in favour of the 262144 native window -- the served-contract change c24 named, chosen for 46.2 tok/s on code (4.65x the floor) and 2.92x concurrency; (2) free-form prose decode drops from ~16.7 to ~13.7 tok/s versus MTP, an accepted trade for structured/tool work. Classified NEEDS-FOLLOW-UP because the deployed 0.57.2 scaffold HARDCODES --speculative-config: this adoption survives only as a hand-edit to the deployment's docker-compose.yml and would be silently reverted by any 'lobes init' re-render. Making it durable requires either re-scaffolding to the current template (where `PRIMARY_SPECULATIVE_CONFIG` is live) or carrying the `speculative_config` as repo data in the shape/catalog.

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|------------------------|-----------------|
| `t6` (`d1`) | Operator request during the live run. It is not merely a memory-headroom question: at 786432 the TARGET is YaRN-extended while the 1.36B drafter declares only 262144 positions, so the drafter is being asked to operate beyond its trained/declared range — a plausible acceptance-suppressing confound distinct from the FP8-vs-W4A4 mismatch already recorded as c20/r1. At 262144 no YaRN extrapolation is asked of either model, KV demand drops far below the 40.76 GiB ceiling that refused the first boot, and DSpark is measured in the regime it was actually built for. Cost: one additional lane restart inside the already-open cortex-down window. | `acceptable` |
| `t6` (`d2`) | Operator request, and it is the sharpest available test of the c20/r1 hypothesis rather than a second opinion on it. vLLM's DSpark speculator consumes the target's MEAN-POOLED AUX HIDDEN STATES — i.e. ACTIVATIONS. The drafter was SpecForge-trained against a Qwen3.8-27B-FP8 target; the deployed cortex is W4A4, which quantizes activations to FP4. An A16 target keeps activations at 16-bit, so its hidden states should sit numerically far closer to the FP8 target the drafter learned from. If the prose acceptance collapse (28.6%) is caused by activation quantization, THIS is the arm that recovers it. Classified RISKY because it violates frame boundary c12 (the deployed checkpoint does not change): it requires serving a different target checkpoint, a ~16 GiB download, and it is no longer a single-variable test against the incumbent. | `risky` |
| `t6` (`d3`) | Operator request. Deviation d1 measured dspark at 262144 but the other two arms were only measured at 786432, so the drafter-aligned window currently has no floor and no incumbent to compare against — exactly the interpretability gap that arm 3 fixed at 768K. c27's own rationale (arm 3 is what makes arms 1 and 2 interpretable) applies per-window, not once globally. | `acceptable` |
| `t6` (`d4`) | Operator decision after reading the five-arm result. t6 was scoped as a SPIKE that ends by restoring the incumbent (c3/h7); adopting a measured arm is a change of deployed serving configuration and therefore beyond that scope. Two consequences are accepted deliberately: (1) the advertised 1M YaRN window is withdrawn in favour of the 262144 native window -- the served-contract change c24 named, chosen for 46.2 tok/s on code (4.65x the floor) and 2.92x concurrency; (2) free-form prose decode drops from ~16.7 to ~13.7 tok/s versus MTP, an accepted trade for structured/tool work. Classified NEEDS-FOLLOW-UP because the deployed 0.57.2 scaffold HARDCODES --speculative-config: this adoption survives only as a hand-edit to the deployment's docker-compose.yml and would be silently reverted by any 'lobes init' re-render. Making it durable requires either re-scaffolding to the current template (where `PRIMARY_SPECULATIVE_CONFIG` is live) or carrying the `speculative_config` as repo data in the shape/catalog. | `needs-follow-up` |

> **Terminology note (raised in review — Qodo PR #200, finding 6).** The `d4`
> record quoted above says "2.92x concurrency". That figure is vLLM's own
> **KV-capacity ratio** — KV pool tokens divided by the declared window, i.e. how
> many requests each occupying the FULL window would fit. It is a modelled
> capacity estimate, **not observed concurrent throughput**: every measurement in
> this run was single-stream batch-1, and `PRIMARY_MAX_NUM_SEQS=2` caps the
> scheduler at two sequences regardless of KV headroom. The deviation records are
> quoted verbatim by contract, so the wording inside them is left as written; read
> "concurrency" there as "KV-capacity ratio". See the Remaining Work entry on the
> absent concurrency measurement.

## Evidence

- tests: `uv run pytest -n auto -q` — 3140 passed, 15 skipped (skips pre-existing, live/optional-dependency gated)
- tests: `tests/test_spike_preflight.py` (22) — pass · `tests/test_spec_arms.py` (17) — pass · `tests/test_catalog.py` (78) — pass
- lint: `uv run flake8 lobes tests` — clean · `uv run black --check lobes tests` — clean · `markdownlint-cli2` on every touched md — 0 errors
- rubric: `uv run afi cli doctor . --strict` — PASS
- commits: `e838061..dde9bf4` (14 commits on `dspark-speculation`)
- evidence transcript: `docs/evidence/2026-08-24-spike-dspark-cortex-spark.txt` (286 lines, 13 sections)
- raw run data: `docs/evidence/arm-*.json` (9 transcripts)
- live restore proof: `scripts/spike-preflight.sh restore` — PASS (healthy + argv intact + live 200 `'awake'`)

## Delivery Claims

| Claim | Confidence | Evidence |
|-------|------------|----------|
| DSpark block-speculative decoding LOADS AND SERVES against a W4A4 NVFP4 target in vLLM `0.26.1rc1.dev942+g5a4c8d992` — no SGLang, no checkpoint swap, no code change | high | transcript §7 (`Resolved architecture: DSparkDraftModel`) · 5 measured arms |
| On the deployed cortex @262144, DSpark decodes 46.20 tok/s on structured/code vs the incumbent MTP's 24.69 | high | `docs/evidence/arm-dspark-262k.json`, `arm-mtp-n2-262k.json` |
| DSpark is a NET LOSS on free-form prose vs MTP (13.71 vs 16.65 tok/s) but stays above the no-speculation floor on every shape | high | `arm-dspark-262k.json`, `arm-mtp-n2-262k.json`, `arm-none-262k.json` |
| The no-speculation floor is ~9.9-11.4 tok/s and flat across content shapes | high | `arm-none-768k.json`, `arm-none-262k.json` |
| The 1.36B drafter costs ~35% of the KV pool and makes the 1M window infeasible at util 0.58 (vLLM's own estimated ceiling: 824000) | high | transcript §5 (the `ValueError`) · KV pool 1,274,831 -> 830,827 |
| The prose acceptance collapse is NOT caused by context extrapolation (d1) | high | `arm-dspark-262k.json` vs `arm-dspark-768k.json` — unchanged within noise |
| The prose acceptance collapse is NOT caused by activation quantization (d2) | medium | `arm-a16-dspark-262k.json` — acceptance within ~1 point of W4A4 on every shape; two figures exactly equal, so the log-derived surface may be coarse (transcript §12 caveat) |
| The deployed 0.57.2 scaffold HARDCODES `--speculative-config`, making `PRIMARY_SPECULATIVE_CONFIG` and the documented off-switch dead knobs on that box | high | transcript §6 · caught by `scripts/spike-preflight.sh` argv proof |
| The Spark cortex now serves DSpark @262144 and is healthy (d4 adoption) | high | `spike-preflight.sh restore` PASS · argv proof shows the dspark token + `--max-model-len=262144` |
| Single-run measurement variance is ~±10-13%; the earlier "window trade-down is worth +10-16%" claim is WITHDRAWN | high | transcript §13 (non-monotonic across 1M/768K/262K; two floors 13% apart) |
| Block speculation is lossless (output equivalence preserved) | unverified | not measured — structural property of the technique only; no output-equivalence test was run |
| The FP8-trained-drafter vs W4A4-target mismatch explains the prose collapse | unverified | d2 removed only the activation-quantization explanation; the hypothesis itself is untested |

## Remaining Work / Follow-up

- **`d4` is a HAND-EDIT and is revert-prone.** The adoption lives in the deployment's `~/.lobes/docker-compose.yml`, not in repo data. Any `lobes init` re-render silently reverts the cortex to MTP n=2 — with no error and no visible symptom except a speed drop. **Next step:** either re-scaffold the deployment to the current 0.59.0+ template (where `PRIMARY_SPECULATIVE_CONFIG` is a live knob) or carry `speculative_config` for cortex as repo data in `lobes/profiles/builtin_shapes/spark-lobe.toml` / the catalog. Until one of those lands, treat the deployed config as fragile. An inline warning comment sits next to the edited line.
- **The 1M window is withdrawn from the served contract.** `lobes capabilities` / `GET /capabilities` now advertise 262144 for cortex. Consumers given the near-1M streaming guidance (2026-08-19 rollout notes) should be told. **Next step:** notify mesh consumers; decide whether `docs/qwen3.8-27b-nvfp4.md`'s 1M validation section needs a deployed-vs-validated split beyond the caveat t7 added.
- **`lobes/_metrics.py` maps no spec-decode/acceptance fields** for either engine, so acceptance is only observable by reading vLLM's `/metrics` or its logs directly. **Next step:** optional follow-up to map the `vllm:spec_decode_*` counters.
- **`num_speculative_tokens` was 7 throughout; `dspark_draft_topk` was never set.** A lower block size may suit prose better. UNMEASURED. **Next step:** a cheap sweep inside one cortex-down window.
- **No concurrency measurement was taken** — every figure is single-stream batch-1, and `PRIMARY_MAX_NUM_SEQS=2` caps the scheduler regardless of KV headroom. **Next step:** a batched run if concurrent throughput matters.
- **No repeat-variance figure.** One run per configuration; ±10-13% is inferred from cross-config disagreement, not measured directly. **Next step:** repeat one arm 3x to quantify it.
- **The `agent/t3`, `agent/t4`, `agent/t5` branches** from an earlier, unrelated workforce run still exist and were deliberately left untouched. **Next step:** operator to confirm they are spent, then delete.
