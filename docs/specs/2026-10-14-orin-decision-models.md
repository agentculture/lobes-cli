# orin decision models

> The Jetson AGX Orin serves the mesh's decision models — Cloudflare Clef 27B (schema-scored decisions, Ampere-fit 4-bit) and, later, a fine-tuned Strands Decider 2B — through lobes, addressable from every mesh member

## Audience

- Mesh agents and harnesses (culture agents, Qwen Code, reachy-mini-cli, lobes route) that need a fast, calibrated multiple-choice decision — routing, tool selection, triage, guardrails — instead of parsing a chat model's prose; and the operator deploying it.

## Before → After

- Before: Today the fleet has no decision model: every typed decision is a chat completion whose answer must be parsed out of prose (lobes route asks the hand model for JSON, lobes/cli/`_commands`/route.py), with no per-option probabilities, and no lobes surface knows /v1/systemone.
- After: Any mesh member answers model=decider on POST /v1/systemone: the Orin serves Clef 27B (4-bit, co-resident with orin-embed) and any box may host a Strands Decider 2B; GET /capabilities advertises the decider role with its checkpoint, quant and feasibility, and a box that hosts none forwards to a verified member.

## Why it matters

- A schema-scored decision returns calibrated probabilities over a closed option set in one forward pass — no generation, no output parsing, no hallucinated option — so callers can threshold on confidence and escalate to cortex only when the decider is unsure.

## Requirements

- Clef is served through a decision endpoint (the card's documented route is SGLang exposing a Jev/SystemOne-compatible POST /v1/systemone with state + questions), not the OpenAI chat surface; a plain 'vllm serve' never runs the joint head, so lobes needs a new non-chat lane type the gateway can route — nothing under lobes/ or docs/ mentions systemone today.
  - honesty: The gateway proxies POST /v1/systemone to the decider lane byte-for-byte (no reshaping) and 404s `role_infeasible` when no decider is hosted or reachable.
- Clef's 4-bit checkpoint on the Orin (Ampere `sm_87`) must be WEIGHT-ONLY W4A16 through Marlin: W4A4 NVFP4 needs Blackwell FP4 tensor cores and is ruled out (lobes/machines/orin.py:10-30) and FP8 W8A8 is unsupported on Ampere. Lead candidate JRamirez-UAB/clef-GPTQ-W4A16-g64-embed-int8 (compressed-tensors, ~18 GB, head/ViT/`lm_head` bf16, vLLM 0.31.0 + a 4-line `qwen3_5` embed-quant patch + `clef_vllm.py` pooling wrapper, 65/68 match vs bf16); second AutomatosX/AX-Clef-27B-CUDA-AXQ-NVFP4-W4A16; avoid kurcontko/simonlehmann clef-NVFP4 (W4A4) and prithivMLmods/clef-FP8.
  - honesty: The served Clef build's config.json `quant_method` is weight-only (W4A16/WNA16 or NF4) and vLLM/transformers logs show a Marlin/weight-only kernel, never an FP4-activation path, on `sm_87`.
- Adding the 'decider' role touches every per-role surface: ROLES + `ROLE_BACKEND`/`ROLE_ROLE_HINT`/`ROLE_PATH`/`ROLE_RESPONSIBILITIES`/`ROLE_FORBIDDEN`/`ROLE_MAX_MODEL_LEN_ENV` (lobes/roles.py:113-559), `COLLEAGUE_ROLES`/`OPT_IN_CORE_ROLES` (lobes/profiles/shapes.py:92,114), schema.py role scope, render.py `ROLE_ENV_PREFIX`, `shape_render.py` + cli/`_commands`/up.py `ROLE_SERVICE`, gateway/`_config.py` `OPT_IN_BACKENDS`/`FEASIBLE_ENV`/\*`_BASE_URL`, the fleet compose template, every builtin profile + shape TOML, and the ~28-40 golden .env files (tests/goldens/regen.py) — plus a /v1/systemone route the gateway proxies and mesh-forwards (no 'decider' name collides today; only a comment at lobes/roles.py:474 uses the word).
  - honesty: tests/goldens/regen.py regenerates every golden with a decider line and the full suite passes; docs/colleague-stack.md records the decider role's justification for the irreversible add.
- decider role contract: responsibilities `typed_decision`, classification, `routing_choice`, triage, `guardrail_check`, `final_decision`; forbidden `repo_action`, `security_decision` — it decides over a closed option set and returns the best answer + all probabilities, but never acts on the repo or rules on security.
  - honesty: lobes/roles.py `ROLE_RESPONSIBILITIES`/`ROLE_FORBIDDEN` for decider match this text and tests/`test_colleague_contract.py` pins it.
- Domain routing: each decider:<domain> is advertised in GET /capabilities with its checkpoint and quant and mesh-forwarded by name (like a specialist lane, docs/colleague-stack.md#specialist-lanes-not-roles); a domain name must be mesh-unique — two members announcing the same domain with different fingerprints is a refused conflict, never a silent pick; an unknown domain 404s `model_not_found` (mirrors hand:<domain>).
  - honesty: A test registers two members announcing the same domain with different fingerprints and asserts the conflict is refused; a cross-box decider:<domain> forward is shown in an evidence transcript.

## Honesty conditions

- Clef answers model=decider on the Orin through the gateway with orin-embed's three lanes still serving, and a 2B decider answers from at least one other box — both shown in an acceptance transcript under docs/evidence/ (#108), not declared.
- decider's `ROLE_FORBIDDEN` and hand's stay distinct; no decider checkpoint is routed through vllm-hand's --enable-lora lane.
- orin.toml `exclusive_roles` and tests/`test_init_coresidency.py` are updated so a Clef-hosting shape is refused alongside cortex/senses/associate.
- No jev-factory/training code lands in this repo's PRs for this frame.
- At least one named consumer (lobes route or a culture agent) has a concrete decision it would send to /v1/systemone instead of a chat completion.
- grep finds no systemone/decider surface in lobes/ on main at the time of writing (verified 2026-10-14 in the scope pass).
- The per-option probabilities the decider returns are usable for thresholding — the strands/clef cards' calibration claims hold on our own probe set, or the caveat is recorded.
- Every surface named (model=decider, /v1/systemone, /capabilities advert, mesh forwarding) exists and is exercised by a test or an evidence transcript.
- The probe set, the bf16 reference answers and the latency numbers are all filed in a docs/evidence/ transcript; the bf16 reference is computed off-box (Spark) since bf16 Clef cannot load on the Orin.
- The 2B is booted on a box other than the Orin and reached cross-box via the mesh, transcripted.

## Success signals

- On the Orin, with orin-embed still serving: Clef answers a /v1/systemone choice question through the gateway with probabilities summing to 1 (±0.01) per question, top-1 matching the bf16 reference on >= 90% of a fixed probe set, and the minimum available host memory stays >= 2 GiB (orin-embed decision c25 floor); median decision latency for a <= 4096-token state is recorded (sidecar baseline, then the faster path).
- A Strands Decider 2B served bf16 on a second box (e.g. spark2) answers model=decider:<domain> from every other mesh member, with X-Lobes-Mesh-Member on the reply.

## Scope / boundaries

- Neither model is the 'hand' role: hand forbids `final_decision` (lobes/roles.py `ROLE_FORBIDDEN`, hand ~:537; docs/colleague-stack.md:58 'never decides') and its LoRA serving is vLLM --enable-lora on a generate lane (lobes/templates/fleet/docker-compose.yml:790-880) — a pointer/schema head is not a generate adapter.
- Clef takes the Orin's single heavy-lobe slot: orin.toml \[\[`exclusive_roles`\]\] (~:117-119) allows one of cortex/senses/associate (orin-cortex/orin-lobe/orin-associate) on 61.3 GiB unified with zero swap, so a Clef-hosting shape joins that exclusive group (and tests/`test_init_coresidency.py`) and cannot co-reside with the GGUF cortex (~33 GiB) or associate (util 0.70).
- The jev-factory work (producing decider training data / retraining) runs in a separate session and is not part of this frame; this frame consumes its output only.

## Non-goals

- lobes does not train the decider: LoRA/adapter production stays out of tree (docs/colleague-stack.md:322-324 — lobes serves adapters, never trains them); the decider's fine-tune is its own training/recipe.sh run (~11 h on one RTX 3090), whose output lobes only serves.
- This frame does not rewire 'lobes route' (lobes/cli/`_commands`/route.py asks the hand model for JSON with `enable_thinking`=false); pointing routing at a decider is a follow-up once a decision lane exists.

## Assumptions

- Cloudflare/clef (Apache-2.0, 27.36B bf16, `Qwen3_5ForConditionalGeneration` post-trained from Qwen3.8-27B, 48 GDN + 16 full-attn layers, ViT present, `mtp_num_hidden_layers`=0) is a DECISION model, not a chat model: a joint schema head over the backbone's final hidden states emits one logit per allowed option per question in one forward pass — no free-form generation (huggingface.co/Cloudflare/clef card + config.json).
- Strands Decider 2B (StrandsAgents/strands-decider-2B-qwen3.5-v1-2610, Apache-2.0, AWS Strands Labs) is a rank-16 LoRA plus a ~1M-param fp32 pointer head on Qwen/Qwen3.5-2B-Base with the LM head discarded; no merged weights ship; served by the strands-decider package ('strands-decider serve', POST /v1/systemone) — the same wire Clef speaks; trained/evaluated at a 4096-token window (strands-labs/strands-decider README, HF card).
- Memory fits beside the deployed orin-embed shape: measured resident embedder 5.24 + reranker 5.35 + gemma2-embed 4.69 GiB with 43,531 MiB available (docs/evidence/2026-10-07-accept-orin-embed.txt:86), so W4A16 Clef (~18 GB weights + KV at a decision-sized window) plus a bf16 decider (~4.5 GiB) are expected to fit — DECLARED until measured (#108), with the page-cache and cgroup `mem_limit` traps of docs/orin-embed-deployment.md:92-114.
- 'decider' is an OPT-IN core role (joins `OPT_IN_CORE_ROLES` like muse/worker/associate): machine-as-brain never hosts a 27B decision model beside cortex+senses, an unwired decider 404s `role_infeasible`, and only an explicit Orin shape hosts it.

## Scope exploration

- `s1` — `huggingface.co/Cloudflare/clef (card + config.json)`: decision model with joint schema head; SGLang /v1/systemone documented route; bf16, no MTP, ViT present
  - seeds: `c2`, `c3`
- `s2` — `HF quantized children of Cloudflare/clef`: 35 children; W4A16 GPTQ (JRamirez, vLLM 0.31.0 + patch + wrapper) and NVFP4-W4A16 (AutomatosX) are Ampere-viable; NVFP4 W4A4 and FP8 are not
  - seeds: `c4`
- `s3` — `lobes/machines/orin.py + docs/evidence/2026-09-13-measure-associate-budget-orin-1m.txt`: `sm_87` runs only weight-only paths via Marlin (MarlinLinearKernel / WNA16); W4A16 compressed-tensors proven on Lightning at 1M
  - seeds: `c4`
- `s4` — `strands-labs/strands-decider README + StrandsAgents HF cards`: LoRA r16 + fp32 pointer head on Qwen3.5-2B-Base; no quantized official variants; serve via package on /v1/systemone; retrain via recipe.sh
  - seeds: `c5`, `c6`, `c8`
- `s5` — `vllm-project/vllm#29424 + docs.vllm.ai LoRA page`: LoRA on GPTQ/AWQ/compressed-tensors works for MoE test models; no statement for GDN hybrids; GGUF/bnb unsupported
  - seeds: `c6`
- `s6` — `lobes/roles.py + lobes/templates/fleet/docker-compose.yml vllm-hand`: hand forbids `final_decision`; its LoRA path is a vLLM generate lane — not a fit for schema-scored heads
  - seeds: `c7`
- `s7` — `docs/colleague-stack.md:322-324,337-365`: training is out of tree; adding a role is effectively irreversible
  - seeds: `c8`
- `s8` — `lobes/profiles/builtin/orin.toml + builtin_shapes/orin-*.toml`: one heavy lobe via `exclusive_roles`; orin-cortex = llama.cpp Qwen3.8 GGUF `Q4_K_M` at ~2.4 tok/s, UNVALIDATED
  - seeds: `c9`
- `s9` — `docs/evidence/2026-10-07-accept-orin-embed.txt + docs/orin-embed-deployment.md`: ~43.5 GiB available on the deployed orin-embed shape; page-cache and `mem_limit` traps
  - seeds: `c10`
- `s10` — `lobes/cli/_commands/route.py`: route asks the hand model for JSON gear choice; a decider is a natural future backend
  - seeds: `c11`
- `s11` — `lobes/catalog.py ENGINE_SGLANG`: SGLang is a catalog engine value only (DSpark target/drafter entries, configured); no compose lane renders it (render.py:270)
  - seeds: `c3`

## Decisions

- Strands Decider 2B is served UNQUANTIZED (bf16 torso ~4.5 GiB + fp32 head): no official AWQ/GPTQ/W4A16/FP8 exists, LoRA over a 4-bit base is unvalidated on a GDN hybrid on `sm_87` (vllm#29424 tested MoE only), the training recipe and adapters are bf16, and 4.5 GiB is small beside the Orin's ~43 GiB headroom. W4A16 (compressed-tensors/GPTQ) is the fallback only if memory forces it, measured first.
- Clef serving is staged: start with a transformers sidecar running Cloudflare's own `load_release_model` (the embed-st sidecar pattern) to MEASURE decision speed on the Orin first; the target path is the faster one — (a) vLLM pooling runner on the W4A16 quant + external joint head, or (b) SGLang dev-clef — chosen once the sidecar baseline exists.
- Decision models enter the fleet as a NEW ROLE named 'decider' (accepting that adding a role is effectively irreversible, docs/colleague-stack.md:337-365) — not a hand variant and not a specialist lane.
- Clef co-resides WITH the deployed orin-embed specialist (embedder + reranker + gemma2-embed stay hosted) if it fits — a Clef-hosting Orin shape adds decider to orin-embed's hosted set rather than replacing it; fit is decided by a live measurement against the ~43.5 GiB available, not by arithmetic.
- The Strands decider is retrained on the DGX Sparks (spark / spark2); lobes only serves the resulting LoRA + head.
- The decider role's contract is the upstream /v1/systemone API as provided: a request carries a state plus typed questions (noul / choice / score) and the answer is the best option PLUS the probability of every allowed option — lobes passes that wire through, it does not reshape it into OpenAI chat.
- The Strands Decider 2B may be hosted on ANY lobe box (Spark, spark2, Thor, Orin), for various tasks — it is not pinned to the Orin; Clef 27B is the Orin's decider.
- Decider addressing is model=decider:<domain> — one FLAT, mesh-wide namespace of domains (not per-box member lanes): a caller names the domain, never the box, and whichever member hosts that domain answers.
- A box hosts zero, one or more Strands Decider 2B checkpoints (one per domain); none is required anywhere.
- The decider role MAY carry `final_decision` (c27's 'forbidden: `final_decision`' is rejected).
- The target serving engine for the faster stage is vLLM 0.31 (the version the JRamirez W4A16 quant was validated on) — preferred, to be validated on the Orin; expected to work.
- Loading a 4-bit Clef together with its joint head under transformers on the Orin is confirmed as the first spike (the sidecar stage's go/no-go).
- Clef is the general-purpose domain: plain model=decider (no domain) resolves to decider:general, served by Clef 27B on the Orin.

## Open parks

- [unknown_nonblocking] Clef decision latency on the Orin: one forward pass, so GDN decode speed (2.4 tok/s on llama.cpp) is irrelevant, but prefill of up to 16,384-token states on `sm_87` is unmeasured (Lightning measured ~1,612 tok/s prefill, the GGUF 27B ~64 tok/s)
- [unknown_nonblocking] Whether multimodal (image/video) Clef decisions survive the W4A16 quant on the Orin — the JRamirez and kurcontko cards say untested/rejected
- [unknown_nonblocking] Whether the decider's fp32 pointer head + LoRA can run inside vLLM (pooling + plugin) or must stay on the strands-decider package (unauthenticated, binds 127.0.0.1 per research — README does not document binding)

## Resolved vagueness

- [unknown_blocking] vLLM version bar: Clef quants were tested on vLLM 0.31.0 / 0.28.x / 0.23.1-nightly, while the Orin's associate lane pins vllm/vllm-openai:v0.27.1 and the fleet nightly digest's version is unrecorded; an aarch64 + `sm_87` image at >=0.31 is unconfirmed — resolved: Target vLLM 0.31 (validated build of the W4A16 quant); validated on the Orin in the plan.
- [unknown_blocking] The (c) sidecar baseline's memory: Cloudflare's `load_release_model` loads the BF16 checkpoint (~55 GB, 27.36B params) — beyond the ~43.5 GiB available beside orin-embed and near the Orin's 61.3 GiB total with zero swap. The sidecar must load a 4-bit build instead: a transformers-native one (bitsandbytes NF4: Aikimi/clef-nf4, jlancaster/clef-27b-nf4 — unread) or the compressed-tensors W4A16 via transformers; whether the joint head loads over either is unverified — resolved: First plan task: spike a 4-bit Clef + joint head under transformers on the Orin.
