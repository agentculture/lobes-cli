# lobes unifies its generate lane on one vLLM nightly: the Qwen 3.6 27B primary migrates onto the nightly image the Gemma 4 12B gear already runs, both serve with MTP speculative decoding, and lobes publishes a same-engine head-to-head of the two gears decode/prefill throughput and MTP draft acceptance

> lobes unifies its generate lane on one vLLM nightly: the Qwen 3.6 27B primary migrates onto the nightly image the Gemma 4 12B gear already runs, both serve with MTP speculative decoding, and lobes publishes a same-engine head-to-head of the two gears decode/prefill throughput and MTP draft acceptance

## Audience

- lobes operators/maintainers on the DGX Spark GB10 and the Culture mesh that consumes the generate lane (main = 27B primary, multimodal = Gemma 12B)

## Before → After

- Before: the 27B primary runs on stock vLLM 0.19.0 (nvcr.io/nvidia/vllm:26.04-py3) with qwen3_5_mtp MTP (~19 tok/s, 72-79% accept); the Gemma 12B gear runs on vLLM nightly (0.23.1rc1) but with NO spec-decode (~23 tok/s); the two gears run different engines so no fair head-to-head exists and Gemma has no measured MTP win
- After: the whole vLLM-served fleet runs the same nightly vLLM: the 27B primary default image is flipped to nightly (still serving qwen3_5_mtp MTP), the Gemma 12B gear (already nightly) gains a committed MTP verdict, and the embed + rerank (and opt-in minor/14B) gears also move to nightly and keep serving their task family; plus a committed same-engine head-to-head benchmark of the two generate gears (decode/prefill tok/s + MTP accept), inform-only

## Why it matters

- a head-to-head is only trustworthy if both gears run the SAME vLLM engine; unifying on nightly also finally moves the proven primary off the frozen 0.19.0 (the 0.19-to-0.23 jump the 27B doc said was blocked on the 26.04 base, now unblocked by the nightly base image the Gemma gear proved on the GB10) and ends the split-engine fleet; and Gemma per-stream speed gap either closes or is proven unclosable

## Requirements

- the 27B primary default fleet image flips from nvcr.io/nvidia/vllm:26.04-py3 (0.19.0) to the nightly, and it still SERVES with qwen3_5_mtp MTP active (not merely loads): Qwen3_5ForConditionalGeneration + Qwen3_5MTP resolve, /health 200, a tool-call round-trip passes, and draft acceptance is >0
  - honesty: on the nightly image the 27B loads with Qwen3_5ForConditionalGeneration + Qwen3_5MTP resolving, /health 200, a qwen3_coder tool-call round-trip passes, and MTP draft acceptance is >0 (target >= the 0.19.0 baseline 72-79%); gibberish, lost MTP, or a rejected quant is a STOP, not a ship
- the embed (Qwen3-Embedding-0.6B, pooling/convert embed) and rerank (Qwen3-Reranker-0.6B, convert classify) gears, plus the opt-in minor (4B bf16) and legacy 14B NVFP4 gears, also move to the nightly vLLM image and each still serves its task family (POST /v1/embeddings returns vectors; /v1/rerank + /v1/score return scores; minor/14B generate) within the GB10 memory budget
  - honesty: each migrated non-generate gear proves its task family live on nightly: embed returns a correct-dim vector, rerank returns ordered scores, and (when enabled) minor/14B answer a generate probe — all within the ~0.69 fleet budget on the 128 GB GB10
- the Gemma MTP wiring flows through the catalog: the gemma entry speculative_config drives the compose --speculative-config items (the same mtp_compose_command_items pattern the 27B uses), and the three guard tests that assert no-gemma-spec (test_gemma_has_no_speculative_config, test_fleet_compose_multimodal_vision_active_no_spec_decode, the MTP-items drift guard) are flipped to the spec-enabled invariant
  - honesty: lobes switch adds/removes the gemma --speculative-config compose items the same way it does for the 27B primary, the catalog compose-items round-trip test passes, and the three renamed guard tests now assert the spec-enabled invariant (green)
- the head-to-head benchmark runs both generate gears on the same nightly vLLM engine version and records decode tok/s, prefill tok/s, and MTP draft acceptance for each, with the method (co-resident vs standalone, util, max-model-len) recorded
  - honesty: the benchmark artifact records both gears numbers under the SAME nightly vLLM version string, states util/max-model-len/co-resident-vs-standalone method, and flags the 12B-vs-27B size difference so the numbers are not misread as a like-for-like model comparison
- the Gemma MTP verdict is committed via the DSpark draft_model route measured FIRST: restore the catalog speculative_config + compose items as default if it beats the ~23 tok/s no-spec baseline, else document the measured negative
  - honesty: DSpark loads on the serving gemma checkpoint with no vocab/tokenizer mismatch and yields >0 percent draft acceptance; exactly one outcome is committed (restore speculative_config as default, or a documented negative with the measured numbers)

## Honesty conditions

- the shipped state is verifiable in-repo: every vLLM gear serves on the nightly image, the 27B keeps qwen3_5_mtp MTP, the Gemma gear carries a committed MTP verdict, and a same-vLLM head-to-head table exists
- the audience is real, not hypothetical: lobes operators run this fleet on the GB10 and the Culture mesh actually addresses model=main and model=multimodal (verifiable from gateway routing + culture.yaml)
- verified in-repo: the 27B catalog/compose pins nvcr.io/nvidia/vllm:26.04-py3 today and the gemma gear pins the nightly image; the 27B doc records ~19 tok/s + 72-79 percent MTP accept and the gemma doc records ~23 tok/s no-spec
- the generate lane (main + multimodal) has real mesh traffic worth benchmarking — not a dormant tier
- the after-state is checkable in-repo: docker ps / lobes status shows every vLLM gear on the nightly image, and the head-to-head table + the Gemma MTP verdict are committed files
- the scope line holds: the diff touches vLLM gear images + the gemma catalog speculative_config + the benchmark only; it does NOT touch the realtime Parakeet/Chatterbox sidecars or train a draft head
- each of the four success artifacts is a concrete committed check (a passing task-family smoke, a lobes status line, a verdict doc section, a benchmark table) — not a subjective claim

## Success signals

- four committed artifacts: (1) every vLLM gear (primary, multimodal, embed, rerank, and opt-in minor/14B when enabled) serves on the nightly image and passes its task-family smoke (generate + tool-call, embeddings, rerank/score); (2) the 27B primary keeps qwen3_5_mtp MTP with acceptance >0 (target >= its 0.19.0 baseline) and tok/s >= ~19; (3) the Gemma gear carries a committed keep/drop MTP verdict with recorded acceptance percent and baseline-vs-spec tok/s; (4) a committed same-vLLM head-to-head table for the two generate gears (decode + prefill tok/s + accept percent), inform-only

## Scope / boundaries

- in scope: migrate ALL vLLM-served gears (27B primary, Gemma 12B multimodal, embed, rerank, opt-in minor/14B) onto the nightly vLLM backend, wire+measure Gemma DSpark MTP through the catalog, and the same-engine generate head-to-head. out of scope: training a native gemma4_assistant draft head (separate #75 follow-up), and the realtime /v1/audio sidecars (Parakeet STT + Chatterbox TTS are NOT vLLM, so they stay put)

## Non-goals

- NOT auto-promoting Gemma to the generate primary off the benchmark (inform-only, no swap); NOT building a native gemma4_assistant draft head in this effort (DSpark-first, escalate only if DSpark wins-but-insufficient, per #75)

## Assumptions

- the nightly vLLM (0.23.1rc1) proven for gemma4_unified on the GB10 also loads+serves the 27B NVFP4-MTP checkpoint (modelopt quant, Qwen3.5 hybrid Mamba/linear-attention arch, qwen3_5_mtp draft) — NOT yet proven for Qwen3_5; load-bearing for the whole same-vLLM premise
- the nightly vLLM still supports the pooling runners the embed/rerank gears need (--runner pooling --convert embed, --convert classify) and the 4B bf16 / 14B NVFP4 quant paths, so the fleet-wide migration does not silently break a task family

## Decisions

- COMBINED effort (user, 2026-07-01): Gemma MTP wiring+measure + the Qwen 27B nightly upgrade + the head-to-head ship as ONE spec, because a fair benchmark requires both gears on the same vLLM
- COMMIT-TO-NIGHTLY (user, 2026-07-01): the 27B primary migrates to the nightly image as its new DEFAULT regardless, unifying the fleet on one vLLM; the benchmark documents the new state (subject to the primary actually serving correctly — see honesty/hard-questions)
- BENCHMARK IS INFORM-ONLY (user, 2026-07-01): it publishes both gears numbers on the same vLLM; no automatic promotion, tier change, or primary swap rides on the result
- DRAFT ROUTE = DSpark-FIRST (user, carried from #75, reaffirmed 2026-07-01): wire+measure deepseek-ai/dspark_gemma4_12b_block7 via the draft_model method first; escalate to native google/gemma-4-12B-it-assistant (mtp, n_predict=1) only if DSpark proves the win real but insufficient
- FLEET-WIDE MIGRATION (user, 2026-07-01): move ALL vLLM-served gears to the new nightly vLLM backend, not just the two generate gears — unify the entire fleet on one engine. Realtime Parakeet/Chatterbox sidecars are excluded (not vLLM). Opt-in minor/14B migrate too but their live-validation may lag the default-on gears

## Hard questions

- risk: nightly vLLM is memory-hungry (proven on GB10 but flagged may need primary down); the whole fleet co-resident on nightly may exceed the GB10 budget, forcing sequential/standalone validation or a trimmed default fleet
- is the benchmark co-resident (both gears live behind the gateway on nightly) or standalone (each on a free host port)? co-resident is the truest same-fleet number but may not fit memory; standalone is the fallback
- risk: commit-to-nightly presumes the 27B serves correctly; a hard functional regression (gibberish, quant rejected, MTP gone) OVERRIDES commit-now and blocks — rollback is the pinned 0.19.0 image, kept available until nightly proves >= parity
- qwen3_5_mtp is already deprecated on 0.19.0 (log: method qwen3_5_mtp is deprecated and replaced with mtp); the nightly may require method=mtp or change the Qwen3.5 hybrid-attention/FLA path — the primary could lose MTP or gibber. Verify at serve time; fallback is method=mtp

## Open / follow-up

- native gemma4_assistant draft (google/gemma-4-12B-it-assistant) as the escalation route if DSpark is insufficient (per #75)

## Accepted unknowns (non-blocking — exporter drops these; restored by hand)

> The devague `spec_md` exporter renders only `follow_up` / `out_of_scope`
> vagueness; these `unknown_nonblocking` items live in the frame JSON
> (`.devague/frames/lobes-unifies-its-generate-lane-on-one-vllm-nightl.json`,
> v1–v3) so `/spec-to-plan` still reads them, but are omitted from the exported
> spec_md — restored here by hand.

- **v1** — the exact nightly image tag/digest to standardize the whole fleet on:
  the gemma gear pins `vllm/vllm-openai:nightly` by digest + the `vllm[audio]`
  extra; do the text-only gears (primary, embed, rerank, minor, 14B) reuse that
  exact digest, or a lighter text-only nightly at the **same version**? (A single
  pinned digest for the whole fleet is the simplest same-engine guarantee.)
- **v2** — DSpark's recommended `num_speculative_tokens` on this serving
  checkpoint (`config.json` `block_size=7`; the disabled experiment used `3`) —
  measured at serve time (t-measure).
- **v3** — per-gear migration order, and whether the opt-in `minor` (4B bf16) /
  legacy 14B NVFP4 gears' live-validation **blocks** this effort or **trails** as
  a follow-up (the four default-on gears — primary, multimodal, embed, rerank —
  are the migration's critical path; minor/14B are opt-in profiles).
