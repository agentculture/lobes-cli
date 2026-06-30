# lobes now defaults to a Spark duo: the Qwen3.6-27B-MTP primary paired with a Gemma 4 12B NVFP4 multimodal worker that takes over the normal/middle tier, replacing the text-only Qwen3-14B (demoted to a legacy candidate profile); the duo gives Spark multimodal senses, a diverse non-Qwen mind, and a DSpark speculative-decoding test target, with old models kept behind explicit legacy profiles.

> lobes now defaults to a Spark duo: the Qwen3.6-27B-MTP primary paired with a Gemma 4 12B NVFP4 multimodal worker that takes over the normal/middle tier, replacing the text-only Qwen3-14B (demoted to a legacy candidate profile); the duo gives Spark multimodal senses, a diverse non-Qwen mind, and a DSpark speculative-decoding test target, with old models kept behind explicit legacy profiles.

## Audience

- Spark's agent stack and the Culture mesh peers that consume the lobes OpenAI endpoint — they gain a multimodal worker and a diverse second mind alongside the Qwen primary

## Before → After

- Before: today the normal/middle tier is a text-only nvidia/Qwen3-14B-NVFP4 that is opt-in (COMPOSE_PROFILES=middle), never load-tested (status=configured), and same-family as the primary — so the fleet has no multimodal generate gear and no non-Qwen mind
- After: lobes serve (and fleet up) bring up the Qwen3.6-27B-MTP 'main' gear + the Gemma 4 12B NVFP4 'multimodal' gear by default; the three live generate gears are addressed as main / minor / multimodal, and model=multimodal reaches Gemma for image+text

## Why it matters

- a multimodal worker lets Spark see screenshots/UI and images (the MTP primary is text-only); a non-Qwen family gives genuinely diverse reasoning for review/triage; and Gemma 4 gives a concrete DSpark/MTP speculative-decoding acceleration target on Blackwell

## Requirements

- Gemma serves WITH its vision tower enabled (NOT --language-model-only) so model=normal accepts image inputs — multimodal coverage is the headline capability
  - honesty: vLLM on the production image serves Gemma 4 12B with the vision tower and returns a correct answer to an image+text request via the gateway model=normal route
- the duo + pooling gears fit the 128GB GB10 unified memory budget with measured gpu-memory-utilization values (Gemma multimodal weights + ViT + image KV vs the 14B's 0.12)
  - honesty: a measured run shows primary + Gemma + embed + rerank (+ minor if kept) all healthy and co-resident under 1.0 total util on the 128GB GB10
- Gemma gets a new role_hint='multimodal' in catalog.py; the old 'middle' role and 'normal' alias are deprecated; the normal tier slot resolves to Gemma for back-compat, and nvidia/Qwen3-14B-NVFP4 is demoted to role_hint=candidate (legacy, kept)
  - honesty: a clean lobes serve with no flags yields exactly two generate gears up (main + multimodal) and zero legacy gears, verifiable via /v1/models and the pressure/status output
- lobes serve no longer means single-model: with no extra flags it brings up BOTH main (Qwen primary) and multimodal (Gemma); minor, the 14B, and every other legacy gear require an explicit profile/override to serve
  - honesty: the default Gemma gear boots with native-MTP speculative-config active (measurable draft acceptance > 0), and flipping the DSpark toggle off is the default state with no DSpark weights loaded
- the default Gemma 'multimodal' gear serves NVFP4 + its native MTP assistant (speculative decoding ON); the DeepSeek DSpark draft (dspark_gemma4_12b_block7) is a separate, disabled-by-default experiment toggle
  - honesty: the default Gemma gear's compose command carries a native-MTP --speculative-config and NO DSpark draft, and a single documented env toggle (off by default) swaps in deepseek-ai/dspark_gemma4_12b_block7
- the default 'multimodal' gear ALSO serves Gemma 4's native AUDIO modality (audio-in understanding / ASR via the Gemma4Unified multimodal embedder, which vLLM supports), so model=multimodal accepts audio content in a chat request alongside text+image — distinct from and NOT replacing the existing /v1/audio/* overlay (Parakeet STT / Chatterbox TTS) per c6
  - honesty: vLLM on our image serves the chosen Gemma 4 12B build with the audio modality active, and model=multimodal returns correct text for an audio-bearing chat request (e.g. a short transcription) without disabling vision or the existing /v1/audio/* overlay

## Honesty conditions

- a fresh lobes install with no overrides serves exactly the Qwen 'main' + Gemma 'multimodal' duo, and every legacy model (14B, minor, etc.) is reachable only behind an explicit profile/override
- a Spark agent / Culture mesh peer reaches the multimodal gear through the existing lobes OpenAI endpoint with no client change beyond sending model=multimodal
- this matches today's shipped state, verifiable in the repo: the 14B middle is opt-in (COMPOSE_PROFILES=middle), status=configured in catalog.py, hermes parser, text-only — and there is no multimodal or non-Qwen generate gear in the fleet
- an image-bearing request the text-only primary cannot serve is answered by the multimodal gear, and the served family is observably Gemma (a different lineage from the Qwen primary/minor)
- after the change the embed/rerank/audio/minor gears and all other catalog entries are unchanged except the 14B role demotion and the new Gemma entry; no lobes train verb is added; DSpark ships off
- a committed smoke test asserts model=main (text) and model=multimodal (image round-trip) both return valid output, and that an explicit legacy profile (e.g. the 14B) still boots and serves
- model=multimodal routes to Gemma at the gateway, model=main routes to the Qwen primary, and the catalog test stays green with role_hint='multimodal' present and the 14B as a candidate

## Success signals

- a smoke test confirms both default models are reachable (model=hard text + model=normal multimodal image round-trip) and that a legacy profile (e.g. the 14B) can still be selected explicitly

## Scope / boundaries

- not removing the Qwen3-14B or any catalog gear (kept as legacy candidates); not changing the embed/rerank pooling gears, the audio overlay, or the minor 4B LoRA base; not adding a lobes train verb; DSpark stays a disabled-by-default experiment

## Open / follow-up

- Gemma 4's native ASR (model=multimodal audio-in) OVERLAPS the Parakeet STT sidecar behind /v1/audio/transcriptions — decide in planning whether Gemma subsumes, augments, or co-exists with Parakeet (the OpenAI /v1/audio/* endpoint shape still needs Parakeet/Chatterbox; Gemma audio-in is chat-only). Bonus: Gemma 4 also ingests VIDEO — out of scope for v1 but available.

## Accepted plan risks (verify during implementation)

> Tracked, non-blocking unknowns carried in the frame (`devague status` → `parked_items`). They are preserved for `/spec-to-plan`, where each becomes a first-class plan risk/validation task. Not blockers — but each must be resolved before its gear is promoted to `load-tested`.

- **Checkpoint pick (v1):** exact HF id for the default Gemma 4 12B NVFP4 gear. NVFP4 12B builds provably exist; leading candidate `sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4` (same publisher as the lobes primary, NVFP4 + native MTP — matches the MTP-on decision); alternatives `AxionML/Gemma-4-12B-NVFP4`, `coolthor/gemma-4-12B-it-NVFP4A16`. NVIDIA ships only 31B + 26B-A4B NVFP4, so the 12B is community.
- **Tool parser (v2):** the vLLM `--tool-call-parser` for Gemma 4 + a `runtime/_parser.py` `infer_parser` rule so the catalog test (`tool_parser == infer_parser(id)`) stays green.
- **Image load (v3):** whether the chosen build loads non-gibberish on the current Blackwell image (`nvcr.io/nvidia/vllm:26.04`) or needs a newer engine — *lower risk*: vLLM registers `Gemma4UnifiedForConditionalGeneration` and auto-detects NVFP4, and Gemma 4 is a mainstream multimodal arch (not the hybrid-FLA arch that gave Qwen3.5 gibberish on sm_120/121). Verify the image engine version before promoting to `load-tested`.
- **Speculative-config (v4):** the exact `--speculative-config` method+JSON for (a) Gemma 4 native MTP (default gear, analogous to the primary's `{method: qwen3_5_mtp}`) and (b) the DSpark experiment draft. DSpark shipped 2026-06-27 via DeepSeek's DeepSpec (`deepseek-ai/dspark_gemma4_12b_block7`, validated on Gemma4-12B, vLLM-servable).
- **GPU budget (v5):** measured `--gpu-memory-utilization` for the *multimodal* Gemma gear (vision + audio embedders + image/audio KV vs the 14B's 0.12) — does 0.12 hold or does the duo need primary/util retuning to fit 128 GB?
