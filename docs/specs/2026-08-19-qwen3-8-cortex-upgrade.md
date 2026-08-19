# qwen3.8 cortex upgrade

> lobes upgrades the cortex checkpoint from unsloth/Qwen3.6-27B-NVFP4 to unsloth/Qwen3.8-27B-NVFP4 on the DGX Spark, served on a vLLM engine new enough to register the Qwen3.8 arch, with measured GB10 budgets and the option of 1M YaRN context
> instruction: follow docs/model-switch-playbook.md ordering: benchmark the incumbent on the current engine first, then bump the shared digest (official nightly first), boot 3.8 at 262144 to isolate checkpoint from YaRN risk, then extend to 1M, then run the correctness gates and land the evidence transcript

## Audience

- the fleet operator plus every mesh consumer that pins the raw served id (culture/colleague, eidetic, reachy-mini-cli, the lobes agent's culture.yaml) and the peer boxes proxying cortex

## Before → After

- Before: cortex serves unsloth/Qwen3.6-27B-NVFP4 at 256K native on the shared vLLM 0.23.1rc1.dev672 nightly digest (spark-lobe 0.44@262144, co-resident with embedder/reranker/embed-deep/hand)
- After: cortex serves unsloth/Qwen3.8-27B-NVFP4 at 1M YaRN context on the DGX Spark via a new shared official-nightly digest with self-hosted MTP armed and image+video intake, all five digest-sharing lanes revalidated, the 3.6 demoted to candidate, and consumers/peer mirrors repointed

## Why it matters

- the fleet's reasoning/final-authority lobe gains a newer checkpoint generation and ~4x served context (256K -> 1M) — long-context work that currently cannot fit in cortex at all becomes servable on the same box

## Requirements

- lobes/catalog.py gains a SupportedModel entry for unsloth/Qwen3.8-27B-NVFP4 verified against its published config.json (arch id, quant scheme, MTP tensors, ViT, 262144 native), following the exact pattern the current primary entry used; the incumbent unsloth/Qwen3.6-27B-NVFP4 demotes to candidate, kept not deleted (cite-don't-delete, same as the sakamakismile demotion)
  - honesty: every fact in the new catalog entry is read from the checkpoint's published config files at implementation time (not from the forum or this spec), and the demoted 3.6 remains selectable via lobes switch
- lobes/profiles/builtin/spark.toml \[primary\].model and lobes/profiles/`builtin_shapes`/spark-lobe.toml flip to unsloth/Qwen3.8-27B-NVFP4 with budgets treated as MEASURED truths per the repo rule: boot the spark-lobe hypothesis 0.44/262144 first (the incumbent's accepted values; the forum measured 0.45/262144 for this exact checkpoint on a GB10, so 0.44 is plausible); spark.toml's machine-as-brain duo 0.30/131072 stays a declared, never-booted-with-this-ViT inheritance unless separately measured
  - honesty: every `gpu_mem_util`/`max_model_len` committed for the 3.8 lane is a value a live GB10 boot actually accepted — declared-only numbers are labelled UNVALIDATED per #108
- lobes/runtime/`_parser.py` `_RULES` gains qwen3.8 / qwen3-8 / `qwen3_8` markers resolving to `qwen3_coder` (tests/`test_parser.py` + the catalog test asserting `tool_parser`==`infer_parser`(id) both extend); the forum's `qwen3_xml` parser name is a newer-vLLM alternative and is NOT adopted while the fleet stays on the 0.23.1 digest
  - honesty: `infer_parser`('unsloth/Qwen3.8-27B-NVFP4') returns `qwen3_coder` (or the deliberately chosen newer parser) and the catalog test asserting `tool_parser`==`infer_parser`(id) passes
- every in-tree surface naming the served primary id updates together: lobes/templates/{docker-compose.yml,env.example} + templates/fleet/{docker-compose.yml,env.example} (including the `VLLM_MODEL`/`VLLM_SERVED_NAME` == `PRIMARY_`\* coherence rule), lobes/roles.py, lobes/gateway/`_config.py`, lobes/cli/`_commands`/whoami.py, lobes/explain/catalog.py, lobes/machines/orin.py, plus the ~20 test files pinning the id (grep audit 2026-08-19 found 40 files)
  - honesty: after the change, grep for the old id finds it only in candidate/demotion/history contexts — no live default, template, or profile still points at 3.6
- the docs/model-switch-playbook.md ordering governs the live swap: benchmark the INCUMBENT on the current engine BEFORE the swap (that baseline is unrecoverable), measure decode via usage.`completion_tokens` never SSE-chunk counts, and apply the forum's three measurement gotchas — unique prefixes (prefix caching masks prefill), fresh images (mm cache masks vision cost), MTP acceptance averaged over multiple DIFFERENT prompts (acceptance is generated-text-dependent)
  - honesty: the incumbent baseline transcript exists and is committed BEFORE the swap lands, measured via usage.`completion_tokens` with the three cache/acceptance gotchas applied
- the swap breaks every consumer pinning the raw id (playbook §2: NO mesh consumer addresses by role — changing `PRIMARY_SERVED_NAME` 404s culture/colleague, eidetic, reachy-mini-cli and the lobes agent's own culture.yaml), and every box proxying cortex must mirror the new served name into its .env (the 2026-08-05 Spark refresh showed a peer swap left `MULTIMODAL_SERVED_NAME` stale and the role sat ready=false) — the rollout needs a consumer/peer update step outside this repo
  - honesty: the rollout notes name every consumer repo and peer .env mirror that must repoint, and the swap PR links them — none discovered post-hoc via 404s
- deliverables follow the repo's promotion convention: a new docs/qwen3.8-27b-nvfp4.md per-model doc, CLAUDE.md served-model paragraph update, an evidence transcript under docs/evidence/ gating any VALIDATED claim (#108), a CHANGELOG entry and version bump (every-PR-bumps rule), and the incumbent's docs kept (cite-don't-delete)
  - honesty: no doc or capabilities surface says VALIDATED without a docs/evidence/ transcript; the PR bumps the version
- the vLLM image updates to a build that serves Qwen3.8-27B-NVFP4 on GB10 (operator-approved 2026-08-19): candidates are a newer official vllm/vllm-openai nightly digest, or the forum's GB10 builds (ghcr.io/spark-arena/dgx-vllm-eugr-nightly nightly-20260801 = vLLM 0.26.1rc1.dev244, or ghcr.io/drowzeys/keys-vllm-027-gb10-qwen38:mtp3-20260813); the new pin lands as a digest, and the c4 blast-radius rule applies — either the shared digest bumps for all five lanes with revalidation, or a `PRIMARY_IMAGE` per-lane knob is added first
  - honesty: the committed pin is a digest whose resolved vLLM version is recorded; the official image was tried first and any fallback to spark-arena is documented with the failure that forced it
- before promotion, strict tool calling is proven on the new engine: the `qwen3_coder_thinking` plugin's structural-tag patch is re-verified (or ported) against the new digest's vLLM source, or superseded by the newer engine's own parser (`qwen3_xml`) if that engine derives the grammar's reasoning flag correctly — a strict:true tool schema with `enable_thinking` on must return a well-formed call, not a 500
  - honesty: a strict:true + `enable_thinking`=true tool call is exercised live on the new engine and returns a schema-valid call; the 0.23-era plugin is not carried forward unverified

## Honesty conditions

- a live GB10 boot serves unsloth/Qwen3.8-27B-NVFP4 at `max_model_len`=1048576 and a gateway chat completion answers 200 — no claim of 'upgraded' before that boot
- the swap PR either bumps the digest for all five lanes with each lane re-probed, or first lands a per-lane image knob — it never moves cortex alone on the shared default silently
- the rollout notes are addressed to these named consumers/peers; none learns of the swap via a 404
- the before-state numbers (digest, 0.44@262144) match what the live box actually serves at swap time, re-checked not assumed
- every element of the after-state is evidenced: 1M boot transcript, five-lane revalidation results, 3.6 demotion in the catalog, repointed mirrors verified 200
- the 1M claim is exercised, not just configured: a long-context request beyond 262144 tokens completes through the gateway
- the transcript is committed under docs/evidence/ in the promoting PR itself, not promised as a follow-up

## Success signals

- an evidence transcript under docs/evidence/ records the live GB10 boot at `max_model_len`=1048576 with the correctness gates passing, MTP acceptance and decode measured against the pre-swap incumbent baseline, and a gateway chat completion answering 200 for model=cortex and the raw new id

## Scope / boundaries

- the shared `VLLM_NIGHTLY_IMAGE` digest is the default image for the primary, embed, rerank, hand and worker lanes (lobes/templates/fleet/docker-compose.yml:43,181,266,324,491,1078) — if the spike proves a digest bump necessary, the blast radius is every lane on it, and there is no `PRIMARY_IMAGE` per-lane override today (only `HAND_IMAGE`/`WORKER_IMAGE` exist); a bump-for-cortex-only would need a new template knob or a full-fleet revalidation

## Non-goals

- no other lane changes: senses/muse/worker/hand/embedder/reranker/audio keep their checkpoints and knobs, the demoted 3.6 stays a selectable candidate, and no lobes train / YaRN-by-default lands — 1M context is a separate decision, not a rider on the swap

## Assumptions

- the pinned fleet nightly (vllm/vllm-openai@sha256:7c5a... = vLLM 0.23.1rc1.dev672) boots Qwen3.8-27B-NVFP4 unchanged: the checkpoint's config.json (fetched 2026-08-19) declares `Qwen3_5ForConditionalGeneration` / `model_type` `qwen3_5`, compressed-tensors mixed-precision, 64 layers, `mtp_num_hidden_layers`=1 — byte-identical arch/quant declarations to the incumbent the digest serves on this GB10 today; the NVIDIA forum's 'requires custom 0.26.1 build, stock has no `sm_121a` NVFP4 kernels' claim conflicts with this fleet's live experience and must be decided by a standalone spike boot (the docs/vllm-nightly-migration.md t2-spike pattern), not believed
- the #93 `preserve_thinking` flag and the `qwen3_coder_thinking` strict-tools plugin (lobes/`vllm_plugins`/) both survive the swap: the 3.8 `chat_template`.jinja carries the `preserve_thinking` variable (verified 2026-08-19, 1 occurrence) and the tool-call format stays the `qwen3_coder` XML family; the plugin only needs re-verification if the engine digest changes

## Scope exploration

- `s1` — `lobes/catalog.py (primary entry, lines ~150-230)`: the current primary entry documents its own promotion recipe: config-file-verified facts, measured-not-declared budgets, promotion gated on a live GB10 boot + evidence transcript; the 3.8 entry must replicate this and demote the 3.6 to candidate
  - seeds: `c2`
- `s2` — `unsloth/Qwen3.8-27B-NVFP4 config.json + forum post 380244 + docs/vllm-nightly-migration.md`: config declares the SAME arch id and quant scheme the pinned nightly already serves; the forum ran a custom vLLM 0.26.1rc1 GB10 build (ghcr.io/spark-arena/dgx-vllm-eugr-nightly) but its stock-has-no-kernels claim is disproven by this fleet serving the 3.6 NVFP4 on the stock digest — engine compatibility is a spike, not a known migration
  - seeds: `c3`
- `s3` — `lobes/templates/fleet/docker-compose.yml (image pins)`: five lanes default to the one nightly digest; `HAND_IMAGE` and `WORKER_IMAGE` are the only per-lane image escapes — a cortex-only engine bump is not currently expressible
  - seeds: `c4`
- `s4` — `lobes/profiles/builtin/spark.toml + builtin_shapes/spark-lobe.toml`: spark-lobe 0.44@262144 was RE-VALIDATED 2026-07-31 for the multimodal 3.6 cortex; forum measured 0.45@262144 for 3.8 with fixed overhead 28.11 GiB + KV 27.56 GiB — same ballpark, so the incumbent knobs are the right first hypothesis
  - seeds: `c5`
- `s5` — `lobes/runtime/_parser.py + tests/test_parser.py`: `_RULES` matches qwen3.5/qwen3.6 id markers to `qwen3_coder` but has no qwen3.8 marker — an unlisted id would silently fall through to the generic qwen3->hermes rule and misconfigure the parser
  - seeds: `c6`
- `s6` — `unsloth/Qwen3.8-27B-NVFP4 chat_template.jinja + lobes/vllm_plugins/`: `preserve_thinking` present in the 3.8 template; strict-tools plugin patches the 0.23.1 engine's structural-tag call site, so it is engine-version-coupled, not checkpoint-coupled
  - seeds: `c7`
- `s7` — `grep -rl Qwen3.6-27B-NVFP4 across lobes/, docs/, tests/`: 40 files name the current id; roles.py, gateway/`_config.py`, whoami.py, explain/catalog.py and machines/orin.py all carry it beyond the obvious catalog/profile/template surfaces
  - seeds: `c8`
- `s8` — `docs/model-switch-playbook.md + forum post measurement-gotchas section`: playbook's two traps (unrecoverable incumbent baseline; SSE-chunk undercount) plus the forum's three cache/acceptance gotchas define the measurement protocol for this swap
  - seeds: `c9`
- `s9` — `docs/model-switch-playbook.md §2 + eidetic memory spark-proxy-advert-refresh-20260805`: consumers send the raw served id read from /capabilities; peer-readiness probes look for the ADVERTISED id in the peer's /v1/models, so stale mirrors show proxied=true/ready=false
  - seeds: `c10`
- `s10` — `docs/ per-model doc convention + #108 evidence rule`: docs/qwen3.6-27b-nvfp4.md and the evidence-transcript gate are the template for what the 3.8 promotion must ship
  - seeds: `c11`
- `s11` — `CLAUDE.md role contract + fleet lane inventory`: the swap is cortex-lane-scoped; the only cross-lane coupling is the shared image digest (s3) and the pressure/tier vocabulary, which is id-agnostic
  - seeds: `c12`
- `s12` — `forum post 380244 (container section) + operator decision`: forum's verified-working config: vLLM 0.26.1rc1.dev244 custom build, --tool-call-parser `qwen3_xml`, --reasoning-parser qwen3, mtp `num_speculative_tokens`=5 recommended (3 optimal TTFT), `FLASHINFER_CUDA_ARCH_LIST`=12.1a + `FLASHINFER_DISABLE_VERSION_CHECK`=1 on GB10
  - seeds: `c13`

## Decisions

- engine strategy (operator, 2026-08-19): bump the SHARED `VLLM_NIGHTLY_IMAGE` digest for all five lanes that default to it — no per-lane `PRIMARY_IMAGE` knob; embed, rerank, hand and worker revalidate on the new digest
- image source (operator, 2026-08-19): test the newest OFFICIAL vllm/vllm-openai nightly digest first; the third-party ghcr.io/spark-arena/dgx-vllm-eugr-nightly GB10 build is an accepted fallback if the official image cannot serve 3.8 NVFP4 on `sm_121a`
- context (operator, 2026-08-19): serve 1M (`max_model_len`=1048576) via the YaRN hf-overrides on `text_config`.`rope_parameters`; if the co-residency budget does not close at `gpu_mem_util` >=0.60, reclaim GPU memory from the other co-resident gears (trim or drop) rather than giving up 1M

## Open parks

- [unknown_nonblocking] whether the new engine changes the embed/rerank pooling and hand LFM2 lanes' validated behaviour if the shared digest bumps (re-run the three correctness probes)

## Resolved vagueness

- [unknown_blocking] whether the 0.23-engine-patched `qwen3_coder_thinking` strict-tools plugin's call-site patch still applies on a 0.26.x engine (upstream structural-tag code may have moved; `qwen3_xml` parser may supersede) — resolved: converted to requirement c22 (operator accepted it as blocking, 2026-08-19): the plugin re-verification/port is in scope and gates promotion; it is work to do, not an open unknown
