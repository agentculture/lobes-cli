# DSpark speculation on the Spark cortex

> The Spark's cortex got faster without swapping the model or the engine: a DSpark block-speculative drafter now rides the deployed vLLM NVFP4 lane, measured against its own 19.9-24.0 tok/s baseline on the same box — and the repo records SGLang as a known engine axis, so the framework that first demonstrated this is in the history rather than lost to a forum post.
> instruction: Land an evidence transcript under docs/evidence/ named <date>-spike-dspark-cortex-spark.txt covering: the live vLLM DSpark load attempt (success or exact loader error), a same-day incumbent baseline, decode + accepted-tokens-per-step across >=3 content shapes, and the memory delta from the 1.36B drafter.

## Audience

- The lobes fleet operator and the Culture mesh consumers of the Spark's cortex lane — plus the next agent who reads this repo's git history looking for what was tried on GB10 and why.

## Before → After

- Before: Today the Spark cortex decodes at 19.9-24.0 tok/s (measured 2026-08-19, t8) using vLLM's generic self-hosted MTP at `num_speculative_tokens`=2, acceptance 54.4-61.1%. A published SGLang+DSpark recipe reaches 34-38 tok/s on the same GB10 silicon with the same base model, and nothing in this repo records why that gap exists or whether it is reachable from here.
- After: The Spark's cortex serves the same unsloth/Qwen3.8-27B-NVFP4 checkpoint on the same vLLM engine, with a measured decode figure taken under a DSpark speculative config, and an evidence transcript that reports the result either way.

## Why it matters

- cortex is the fleet's reasoning/deciding/final-authority lobe — every role contract routes final decisions through it, so its decode rate is the fleet's thinking speed. A 1.4x-1.6x speedup with no checkpoint change, no engine change and no quality trade (block speculation is lossless by construction — the target model verifies every drafted token) is the cheapest large win available on this box.

## Requirements

- The spike tests whether the DEPLOYED vLLM nightly can drive a DSpark drafter against the deployed NVFP4 cortex. This is not speculative on the engine side: the pinned image (0.26.1rc1.dev942+g5a4c8d992) already declares method 'dspark' — and 'dflash' — in vllm.config.speculative.SpeculativeMethod, probed live in the running container. What is unknown is whether RadixArk/Qwen3.8-27B-DSpark, trained with SpecForge against the FP8 target and shipped for SGLang, loads and accepts against an NVFP4 target in vLLM.
  - honesty: vLLM's dspark method is verified against the ACTUAL pinned image, not a release note — the SpeculativeMethod literal was probed inside the running container, and the load attempt is run before any throughput claim.
- The baseline is re-measured, not quoted. The 19.9-24.0 figures were taken 2026-08-19 on a different .env generation; the run re-measures the incumbent config immediately before the DSpark arm, on the same harness, same prompts, same day — the model-switch-playbook's own rule that the incumbent baseline is unrecoverable afterwards.
  - honesty: The incumbent baseline is captured BEFORE the DSpark config is applied, on the same prompts and harness, and its numbers appear in the same transcript — a quoted 2026-08-19 figure is not accepted as the comparison.
- Throughput is reported ACROSS CONTENT TYPES, not as one number. The published DSpark result is 43-47 tok/s on math/code and 12-18 on free-form prose against a 34-38 headline — a 3x spread driven by acceptance rate. A single-shape measurement of a speculative lane is not a description of it, and cortex-shaped work (reasoning traces, tool calls, prose) spans that whole range.
  - honesty: At least three content shapes are measured (structured/code, reasoning trace, free-form prose), and any headline figure states which shape produced it.
- The SGLang axis is recorded as DATA in the catalog's existing engine field — an `ENGINE_SGLANG` constant alongside `ENGINE_VLLM` and `ENGINE_LLAMA_CPP`, with the RadixArk NVFP4 checkpoint and its DSpark drafter as catalog entries carrying `role_hint`=candidate — so the framework and the recipe survive in-tree even if no lane ever serves them. `serves_with_vllm` already exists as the single predicate that gates vLLM-only behaviour, so a third engine value is an addition, not a refactor.
  - honesty: `ENGINE_SGLANG` lands with tests pinning it the way tests/`test_catalog.py` pins the existing engine set, and no existing lane's behaviour changes — `serves_with_vllm` keeps returning the same answer for every current entry.
- Peers must be checked before the cortex goes down, not just the local box: the Jetson AGX Orin's .env declares `PRIMARY_PEER_ORIGIN`=<http://spark.tail0be7e0.ts.net:8001> — this Spark — from the dual-cortex era, with `PRIMARY_PEER_PROXY` currently false after its 2026-08-23 local-GGUF cutover. A referral origin is a live declaration even when proxy is off, so the run reads each peer's `PRIMARY_PEER_`\* state and reports what pointed here.
  - honesty: Every peer's `PRIMARY_PEER_ORIGIN` / `PRIMARY_PEER_PROXY` state is read and quoted in the transcript before the stop, so 'no peer depended on this lane' is a checked fact rather than an assumption.
- The speculative-config knob is a KNOWN-FRAGILE compose substitution and must be changed with that in mind. docker-compose.yml records that a brace-containing default corrupts compose's interpolation of every later brace pair — the speculative-config default was the measured victim, losing its closing brace — and that --hf-overrides must stay LAST while its value stays single-quoted so the spark-lobe YaRN JSON survives as one argv token. A DSpark config adds a model path plus more braces and spaces to exactly that fragile string, so the run verifies the rendered argv (docker compose config / the container's own command line) BEFORE trusting any measurement.
  - honesty: The transcript shows the actual rendered --speculative-config argv token, not just the .env line — a silently mangled flag can boot a lane that is quietly serving without speculation at all.
- The drafter is UNTRUSTED CODE-ADJACENT WEIGHT running inside the cortex container, from a publisher this repo has never pulled before (RadixArk). It is pinned by revision — the SGLang recipe pins its target checkpoint to 52d1adc5f38aa5ebf099c29ed7025ba34cfbb854 and this repo's convention is digest/revision pinning everywhere — and the catalog entry records the pin, not a floating main.
  - honesty: Any catalog entry or .env value naming the drafter carries an explicit revision, and the transcript records the resolved commit actually loaded.
- If the drafter can only be funded by trading the 1M window down, that is a SERVED-CONTRACT change, not a tuning knob: `max_model_len` is advertised through lobes capabilities and GET /capabilities, was validated live 2026-08-19 with a 328K needle beyond the native ceiling, and consumers were given operator guidance about streaming near-1M requests. Any window reduction is announced with the same weight as the cortex stop, reported in the transcript, and reverted with it.
  - honesty: The transcript states the `max_model_len` in force for every number it reports, so no DSpark figure is ever compared against a baseline taken at a different window.
- The run has a stated ABORT condition, because a spike on a live mesh lane can fail in a way that leaves the fleet worse than it started: if the cortex lane does not return healthy on the incumbent config within a bounded attempt, the run stops, restores, and reports — it does not keep iterating on the DSpark arm with the mesh's reasoning lobe down.
  - honesty: Restoration is verified by lobes status plus one live generate through the gateway, not by container health alone.
- The run measures THREE arms in one cortex-down window: (1) incumbent self-hosted MTP at `num_speculative_tokens`=2, re-measured same-day per c9; (2) DSpark with the RadixArk drafter; (3) speculation fully OFF via the template's documented set-but-empty off-switch with `PRIMARY_MAX_NUM_SEQS` cleared. Arm 3 is what makes arms 1 and 2 interpretable — without it, a weak DSpark result cannot be distinguished from speculation being unhelpful on this hardware generally.
  - honesty: All three arms are measured on the same prompts, the same harness and the same day, and the arm in force is named on every reported number; a missing arm is reported as missing rather than inferred from an earlier transcript.

## Honesty conditions

- The announced speedup is measured on the physical DGX Spark GB10 against a same-day re-measured baseline, with conditions recorded per docs/measuring-lane-performance.md rule 3 — or the announcement is not made.
- The transcript is written to be readable by someone who was not in this conversation: it names the box, the date, the image digest, the config diff, and what it does NOT establish.
- The lane is returned to a healthy, serving state at the end of the run regardless of the result, and lobes status confirms it.
- The 19.9-24.0 baseline is cited as a DATED measurement from a named transcript, never as the lane's current rate — and c9's same-day re-measurement is what any comparison actually uses.
- The 'lossless by construction' claim is stated as a property of block speculation (the target verifies every drafted token), not as something this run measures — and if the spike observes any output difference, that claim is retracted rather than defended.
- At the end of the run, git diff shows no change to `PRIMARY_MODEL`, no change to the engine, and no role re-pointed — only the speculative-config knob and whatever budget it forced.
- The prior frame and its 12 scope entries stay in-tree unmodified; this frame cites the GGUF numbers as published third-party measurements, never as ones this repo took.
- The stop announcement precedes the stop in wall-clock order and is visible in the transcript, and lobes status shows the lane healthy again before the run is called done.
- All three outputs appear in the transcript even when the first is a NO — a failed load reports the exact error text, the incumbent baseline it was compared against, and the memory state at the time.
- The recorded SGLang recipe cites its source (the NVIDIA forum thread and the RadixArk cards) and is marked UNVALIDATED-by-this-repo, since no lane here has run it.
- If measured acceptance lands materially below RadixArk's claimed 3.39 tokens/step, the transcript reports it as a TARGET-MISMATCH hypothesis with the quant difference named — never as a verdict on DSpark as a technique.
- Any acceptance figure names the surface it was read from (vLLM /metrics field or the specific log line), so a later reader can reproduce the reading without guessing.

## Success signals

- The frame succeeds when the transcript answers three things under named conditions: (1) does vLLM load the DSpark drafter against this NVFP4 target at all — yes/no with the exact error if no; (2) if yes, the measured tok/s and accepted-tokens-per-step across at least three content shapes, against a same-day re-measured baseline; (3) what it cost in memory, since the drafter is 1.36B of bf16 on a box already at util 0.58. A clean NO with the loader error is a success — it closes the cheap route and justifies the expensive one.
- Secondarily: git history contains the SGLang recipe — checkpoint, revision pin, image, flags, drafter, acceptance — whether or not this repo ever serves it.

## Scope / boundaries

- The deployed checkpoint and engine do not change. This frame does NOT adopt SGLang as a serving lane, does NOT swap unsloth/Qwen3.8-27B-NVFP4 for RadixArk's export, and does NOT re-point any role — the whole value of the vLLM route is that it changes ONE thing. If the DSpark arm fails in vLLM, the answer is a recorded negative, not an engine migration.
- The GGUF-on-Spark question is CLOSED by this frame, not carried into it: published GB10 measurements put llama.cpp `Q4_K_M` at 15-18 tok/s against a 16.6 tok/s bandwidth ceiling, below the incumbent lane. The prior frame qwen3-8-27b-gguf-on-the-spark keeps that survey and its 12 scope entries; nothing here re-opens it.
- Stopping vllm-primary is operator-gated: the run announces the stop before the mesh's cortex goes down, and the box returns to the incumbent config by re-rendering the prior shape. Permission was granted conditionally, not standingly.
- Acceptance rate is NOT observable through this repo's own metrics surface: lobes/`_metrics.py` maps no spec-decode, draft, or acceptance fields for either engine. So c16's accepted-tokens-per-step figure comes from vLLM's own /metrics endpoint and boot//runtime logs, read directly — and the run does NOT quietly assume lobes measure would have surfaced it. Whether to add a mapping is a follow-up, not this frame's work.

## Non-goals

- Not a quality evaluation. Block speculation is lossless by construction, so this frame measures SPEED and acceptance only; it does not re-open the quant-quality question that issue 194 owns, and it makes no claim about NVFP4-vs-GGUF output quality.

## Assumptions

- The acceptance risk now has a MECHANISM, not just a provenance mismatch. vLLM's DSpark speculator consumes MEAN-POOLED TARGET AUX HIDDEN STATES at configured layers (`dspark_target_layer_ids` / `target_layer_ids`) and reads `sample_from_anchor` off the drafter's `hf_config`. RadixArk's drafter learned those statistics from a Qwen/Qwen3.8-27B-FP8 target; ours is W4A4 NVFP4. Same architecture and layer indices, numerically different hidden states — so the plausible failure is not a load error but SILENTLY LOW ACCEPTANCE, which looks like 'DSpark is not worth it' rather than 'the drafter is mismatched'.

## Scope exploration

- `s1` — `docker exec model-gear-vllm-primary (live probe of vllm.config.speculative)`: The pinned nightly 0.26.1rc1.dev942+g5a4c8d992 already declares DSparkModelTypes = Literal\['dspark'\] and DFlashModelTypes = Literal\['dflash'\] inside SpeculativeMethod, alongside ngram/medusa/`mlp_speculator`/`draft_model`/suffix/`custom_class` and the Eagle+MTP family. The cheap route is declarable on the engine we already run.
  - seeds: `c8`
- `s2` — `https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark`: Standalone 1.36B bf16 drafter: 5 full-attention layers, GQA 40/8, a rank-256 Markov confidence head choosing draft length dynamically, block size 7 (verification width 8), 262144 positions, trained with SpecForge, claimed mean 3.39 accepted tokens/step (2.71-4.57 by task). Its stated target is Qwen/Qwen3.8-27B-FP8 and its stated engine is SGLang — both differ from this fleet's NVFP4/vLLM lane.
  - seeds: `c8`
- `s3` — `docs/evidence/2026-08-19-accept-qwen38-1m-spark.txt (t8) + ~/.lobes/.env`: Incumbent: 24.0/22.3/19.9 tok/s short/medium/long, TTFT 0.20-0.35 s, MTP n=2 at 54.4-61.1% acceptance, `PRIMARY_GPU_MEM_UTIL`=0.58 at a 1M YaRN window with `PRIMARY_MAX_NUM_SEQS`=2.
  - seeds: `c9`
- `s4` — `lobes/catalog.py (ENGINE_VLLM/ENGINE_LLAMA_CPP, serves_with_vllm, speculative_config)`: The engine axis already exists as a first-class catalog field with a single exported predicate; adding `ENGINE_SGLANG` is an addition to a pinned set, and `speculative_config` is already a per-entry string the compose lane substitutes (`PRIMARY_SPECULATIVE_CONFIG` defaults to method mtp, n=2).
  - seeds: `c11`
- `s5` — `forums.developer.nvidia.com DGX Spark 34-38 tok/s thread + ggml-org discussion 27080`: The 34-38 recipe is RadixArk NVFP4 (revision-pinned) + lmsysorg/sglang:qwen38-27b + DSpark, --mem-fraction-static 0.50, torch-compile, num-continuous-decode-steps 2, single-stream batch-1, 43-47 on math/code and 12-18 on prose; llama.cpp on the same box measures 15 plain / 18.08 with draft-mtp.
  - seeds: `c11`
- `s6` — `challenge pass / adjacent-systems lens: orin:~/.lobes/.env + thor:~/.lobes/.env`: Orin declares this Spark as `PRIMARY_PEER_ORIGIN` (proxy false since its local cutover); Thor serves unsloth/Qwen3.8-27B-NVFP4 locally. The mesh is dual-cortex, so Spark downtime is survivable — but the frame never said so, and c14 only covered the local lane.
  - seeds: `c18`
- `s7` — `challenge pass / missing-counter-evidence lens: vllm/model_executor/models/qwen3_5.py + qwen3_next.py`: SupportsEagle3 + `aux_hidden_state_layers` hooks present on the target arch; raises the prior that the cheap route loads.
  - seeds: `c19`
- `s8` — `challenge pass / hidden-dependency lens: vllm/v1/worker/gpu/spec_decode/dspark/speculator.py + vllm/config/speculative.py`: Speculator docstring: 'DSpark consumes mean-pooled target aux hidden states at the target layers, combined to `hidden_size` via `main_proj`'; config requires `target_model_config`, auto-detects the method from 'dspark' in the draft model name (reference artifact deepseek-ai/`dspark_qwen3_8b_block7`), derives `dspark_target_layer_ids` from `target_layer_ids`, and exposes a `dspark_draft_topk` knob.
  - seeds: `c20`
- `s9` — `challenge pass / operations lens: lobes/templates/fleet/docker-compose.yml lines 150-200, 241`: The brace-interpolation corruption and the quoting requirement are both recorded as MEASURED failures in the template's own comments; `PRIMARY_SPECULATIVE_CONFIG` is the documented victim and the documented off-switch (set-but-empty disables speculation entirely).
  - seeds: `c21`
- `s10` — `challenge pass / observability lens: lobes/_metrics.py`: grep for spec/draft/accept returns nothing; the module maps vllm: and llamacpp: prefixed families only. The frame's headline metric has no in-repo observability path today.
  - seeds: `c22`
- `s11` — `challenge pass / security lens: HF publisher provenance + repo pinning convention`: RadixArk is new to this fleet; vLLM's own reference artifact is deepseek-ai/`dspark_qwen3_8b_block7`, a different publisher again. Every image and checkpoint in this repo is pinned by digest or revision — the drafter must not be the exception.
  - seeds: `c23`
- `s12` — `challenge pass / reversibility + contract lens: lobes capabilities surface + docs/evidence/2026-08-19-accept-qwen38-1m-spark.txt`: 1M is an advertised, live-validated capability with published consumer guidance; the frame's memory park treated it as a private budget dial, which understates what trading it down costs.
  - seeds: `c24`
- `s13` — `challenge pass / lifecycle lens: docker-compose.yml PRIMARY_SPECULATIVE_CONFIG comment block`: The template documents the set-but-empty off-switch as the recorded Thor fallback (no `sm_110` MTP kernel), so a no-speculation arm costs one .env line and is already a supported state.
  - seeds: `c25`
- `s14` — `challenge pass / containment + recovery lens: lobes status, the shape re-render path, and the dual-cortex mesh state`: Recovery is a re-render of the incumbent shape plus a health check; the Thor's locally-served cortex means the mesh is not headless during the window, which bounds the blast radius but was never stated in the frame.
  - seeds: `c26`
- `s15` — `challenge pass / concurrency lens: the vllm-primary lane and its co-resident gears`: Clean pass — one writer, one lane, no shared mutable state between the spike and the co-resident embedder/reranker/hand gears beyond the GPU memory pool itself. Residual risk is budget contention, already covered by the memory park, not interleaving.
- `s16` — `challenge pass / migration lens: .env generations`: Clean pass on schema — 0.59.0 made lobes init merge-only so it never truncates .env, and the spike edits one key. Residual risk is human: a hand-edited .env that a later lobes init re-render does not know about.

## Decisions

- The operator wants this TESTED, not just reasoned about — the frame exists to produce a measurement on the physical Spark, not a recommendation.
- SGLang is to be recorded in this repo as a supported-in-principle engine axis, at minimum in git history, because it is the framework that first demonstrated 34-38 tok/s on this hardware. Recording it is an explicit operator goal, not a side effect.
- COUNTER-EVIDENCE FOUND, in the spike's favour: the target architecture already implements what DSpark needs. vllm/`model_executor`/models/`qwen3_5.py` declares SupportsEagle3 with `set_aux_hidden_state_layers`() and `get_eagle3_aux_hidden_state_layers`(), and `qwen3_next.py` concatenates aux hidden states across layers. The aux-hidden-state extraction path DSpark depends on is implemented for this arch in the pinned build — the frame had treated engine-side support as an open risk beyond the method literal.
- DSpark REPLACES MTP rather than stacking with it — method is a single field, so the arm is dspark-versus-mtp and the incumbent's 54-61% acceptance is the thing being displaced. The documented off-switch (`PRIMARY_SPECULATIVE_CONFIG` set-but-empty, plus clearing `PRIMARY_MAX_NUM_SEQS`) gives a THIRD arm for free: no speculation at all, which is the honest floor both speculative arms are measured against.

## Hard questions

- Does the DSpark drafter's SpecForge training against a Qwen3.8-27B-FP8 target transfer to an NVFP4 (W4A4) target at all, or does the hidden-state mismatch collapse acceptance even if the weights load?

## Open parks

- [unknown_nonblocking] Whether vLLM's dspark implementation expects a drafter in a specific export format/config shape that RadixArk's SGLang/SpecForge artifact does not carry.
- [unknown_nonblocking] Whether a cheaper lever — raising the incumbent MTP's `num_speculative_tokens` above 2 — is worth measuring in the same run; the repo records n=3 as OOM-prone at high context with 41-48% acceptance, but that was a different .env generation.
- [unknown_nonblocking] Whether the 1.36B bf16 drafter (~2.7 GiB) fits alongside a cortex at `gpu_mem_util` 0.58 with a 1M KV pool, or whether the 1M window must be traded down to fund it — the spike's first likely wall, and one of its three reported outputs (c16).
- [unknown_nonblocking] Whether vLLM's dspark path has ever been exercised against a W4A4/NVFP4 target by anyone — the published recipe uses SGLang, and the vLLM reference artifact targets an unquantized 8B. This run may be the first such pairing, which is a reason to expect rough edges rather than a reason not to try.

## Resolved vagueness

- [unknown_blocking] Whether the 1.36B bf16 drafter (~2.7 GiB) even fits alongside a cortex at `gpu_mem_util` 0.58 with a 1M KV pool — the drafter's memory is additional, and the 1M window may have to be traded down to fund it. Unmeasured; likely the first thing the spike hits. — resolved: Downgraded per the operator's confirmation of the recommendation: the drafter's memory fit is what the spike MEASURES, not a precondition for writing the spec. Re-parked as non-blocking below.
