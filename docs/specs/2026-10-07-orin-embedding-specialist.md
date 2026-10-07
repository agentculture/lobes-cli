# orin-embedding-specialist

> The Jetson AGX Orin is the mesh's embedding specialist: it hosts several distinctly-addressed embedding lanes — a compact multimodal one (EmbeddingGemma 2: text, code, image, video, audio in one 768-d space) and a code/retrieval specialist picked by a measured head-to-head — each advertising its concrete vector-space identity, reachable from every mesh member, with no cross-model fallback and the 0.6B default embedder untouched.

## Audience

- Mesh consumers needing semantic vectors beyond the 0.6B default — embeddings-cli / coherence-cli research frames, eidetic opting into a richer space, agents doing code/repo retrieval and image/video/audio search — plus the lobes operator who runs and swaps the Orin's lanes.
  - instruction: check docs name these consumers and the per-consumer opt-in env (`EIDETIC_EMBED_MODEL`, `COHERENCE_EMBED_MODEL`)

## Before → After

- Before: Today the Orin runs only the 0.6B embedder+reranker (Qwen3-Embedding-0.6B / Qwen3-Reranker-0.6B at util 0.06 each); the associate lane has been Exited(1) for ~9 days though `COMPOSE_PROFILES`=associate / `ASSOCIATE_GPU_MEM_UTIL`=0.70 are still set, leaving ~46 GiB host memory available.
- After: The Orin renders the orin-embed shape: EmbeddingGemma 2 (multimodal, 768-d), an 8B code/retrieval embedder chosen by measurement, the 0.6B embedder+reranker, and — if memory allows — Qwen3-VL-Reranker-8B, all resident; each lane is advertised on /capabilities with its vector-space identity and reachable by name from every mesh member's gateway.
  - instruction: on a non-Orin member: GET /capabilities lists every Orin specialist lane with `hosted_by` orin, and POST /v1/embeddings model=<lane> returns X-Lobes-Mesh-Member: orin

## Why it matters

- One small text-only 0.6B space cannot do code retrieval well or see images/video/audio; #291 needs several independently trained semantic frames, and the Orin is idle (associate Exited) with ~46 GiB free — the cheapest box to make the mesh's embedding specialist.
  - instruction: cite the orin docker ps / free -m baseline in the acceptance evidence

## Requirements

- New embedding models land as self-named, opt-in task=embed backends on the embed-deep pattern (own compose service, \*`_BASE_URL` gateway wiring, a model= alias plus the raw checkpoint id) — not as a new Colleague role and not as a tier alias.
  - honesty: No new name is added to lobes/roles.py ROLES; each lane is a catalog entry + gateway backend + compose service.
    - instruction: git diff of the PR shows ROLES unchanged; tests/`test_roles.py` role count unchanged
- Each specialist lane advertises its vector-space identity on /capabilities (and lobes capabilities): concrete checkpoint, native dimension, MRL truncation dims, loaded modalities, normalization, runtime, and the box it is load-tested on.
  - honesty: Each specialist lane's /capabilities entry carries checkpoint, native dimension, MRL dims, modalities, normalization, runtime and tested-on box, and the values match the catalog entry.
    - instruction: contract test in tests/`test_gateway_capabilities.py` + live GET /capabilities in evidence
- EmbeddingGemma 2 (google/embeddinggemma-2) is the Orin's first specialist lane: multimodal embeddings for text, code, image, video and audio in one 768-d space, MRL 128/256/512, 8K context, with full and modular (text-only) loads both measured on the Orin.
  - honesty: EmbeddingGemma 2 is measured on the physical Orin in both the full multimodal load and the text-only modular load (resident memory + latency each).
    - instruction: evidence file has both rows with RSS/available-memory and p50 latency
- Multimodal embedding inputs travel in an honest schema — vLLM's chat-style messages extension on /v1/embeddings (or a distinct endpoint where that cannot carry audio) — never flattened into text.
  - honesty: A multimodal request reaches the model with its image/video/audio parts intact; no code path converts a non-text part to a caption or text.
    - instruction: unit test asserting gateway passthrough of the multimodal body + live image probe with a negative control
- The code/retrieval specialist is assigned only after a measured head-to-head on the Orin — EmbeddingGemma 2 vs Qwen3-VL-Embedding-8B (and a permissively licensed code embedder such as nomic-embed-code) on code/repo retrieval over our own repos — not by model card scores.
  - honesty: The head-to-head corpus is drawn from our own repos (lobes-cli, culture, etc.) with labelled relevance, and every candidate is scored on the same corpus/box.
    - instruction: corpus + script committed; results table lists every candidate on identical queries
- A new built-in shape (orin-embed) declares the Orin's embedding-specialist hosting set and budgets, with goldens, replacing orin-associate as what the Orin renders.
  - honesty: orin-embed renders deterministically with goldens for base/orin/spark/thor and does not host associate; orin-associate stays in-tree (cite-don't-delete).
    - instruction: uv run pytest tests/ -k shape; tests/goldens/shapes/orin-`embed__`\*.env exist
- Every mesh member reaches the Orin's specialist lanes by name through its own gateway (single hop, X-Lobes-Mesh-Member), exactly as roles are mesh-forwarded today.
  - honesty: From spark, spark2 and thor, each Orin specialist lane answers by name in a single hop carrying X-Lobes-Mesh-Member: orin.
    - instruction: curl from each member's gateway; record headers in evidence
- A fine-tunable embedding lane may be hosted: a base whose adapters/fine-tunes are produced out of tree (unsloth-cli / Sentence-Transformers) and served under a new distinct model identity, never overwriting the base's vector space.
  - honesty: A fine-tuned EmbeddingGemma 2 is served under its own model identity and the base EmbeddingGemma 2 lane's outputs are unchanged by its presence.
    - instruction: embed a fixed probe set on the base lane before/after adding the fine-tune; cosine == 1.0
- Specialist embedding lanes become first-class ADVERTISED lanes (not roles): /capabilities carries an entry per wired specialist lane with its fingerprint, so `announcement_from_capabilities` announces it and every mesh member auto-wires and forwards it by lane name — today a self-named backend like embed-deep is invisible to both.
  - honesty: `announcement_from_capabilities` includes specialist lanes from /capabilities, and a mesh member that does not host a lane auto-wires it with no per-lane env typed.
    - instruction: unit test in tests/`test_mesh_`\*; live /mesh/roster on spark lists orin's lanes
- Dropping associate leaves it unhosted mesh-wide (the Orin was its only host), so model=associate 404s `role_infeasible` everywhere: lobes docs mark associate DORMANT/unhosted (like muse), and colleague's opt-in associate seat (`COLLEAGUE_ASSOCIATE_MODEL`, addressed by role name) is told via a tracked issue before the Orin is re-rendered.
  - honesty: A colleague issue exists and lobes CLAUDE.md + docs name associate unhosted before orin-embed is applied.
- The EmbeddingGemma 2 lane never runs float16 (bf16 on the Orin, fp32 fallback) and refuses to return a NaN, inf or all-zero vector — a degraded embedding is an error, not a 200.
  - honesty: A unit test feeds a NaN/zero vector through the lane's guard and gets an error; the running lane's dtype is logged bf16.
- Asymmetric task prompts are explicit: the EmbeddingGemma 2 lane accepts a prompt name (query vs document, CodeRetrieval, etc.) per request, echoes the applied prompt in its response metadata, and the code head-to-head applies each candidate's own recommended query/document instructions.
  - honesty: Two requests differing only in prompt name return different vectors and each response names its applied prompt.
- Every Orin lane is memory-contained: vLLM lanes by `gpu_mem_util` and the sidecar by a container memory cap, lanes boot in a healthchecked order, and the acceptance run records zero OOM kills.
  - honesty: The acceptance transcript records each container's memory cap, RestartCount 0 and no OOM-kill lines in dmesg.
- Rollback is one command: re-rendering orin-associate (lobes init --shape orin-associate --profile orin --apply --force) restores the previous Orin deployment byte-for-byte, and the rollback is exercised once before acceptance.
  - honesty: The evidence shows an orin-associate re-render diffing clean against the pre-change deployment, then orin-embed re-applied.
- The operator can start, stop and probe each specialist lane individually (lobes up/status/assess today are keyed off lobes.roles ROLES, so a non-role lane is invisible to them).
  - honesty: Each specialist lane can be started/stopped/probed by a lobes verb without touching any other lane (dry-run by default).
- The role contract becomes 'every ROLES key is present' rather than 'keys == ROLES': tests/`test_colleague_contract.py`:402/421 change to a subset check plus a lane-key check, and lane names may never collide with a role name or tier alias.
  - honesty: A test refuses a lane name equal to any role, tier alias or role-member lane spelling ({role}-{member}); colleague's superset parser still resolves every seat against a payload carrying lane keys.

## Honesty conditions

- Every lane named in the announcement is served from the Orin, advertised with its identity, mesh-reachable, and the plain embedder still answers with Qwen3-Embedding-0.6B.
  - instruction: acceptance transcript exercises each lane by name from a non-Orin member plus model=embedder
- The before-state is the measured 2026-10-07 baseline, not recalled history.
  - instruction: paste ssh orin docker ps + free -m output into the evidence file's baseline section
- The plain embedder/reranker served id on every box is byte-identical before and after (Qwen/Qwen3-Embedding-0.6B / Qwen/Qwen3-Reranker-0.6B) and the mesh embedder pool keeps orin as an agreeing member.
  - instruction: GET /capabilities embedder.model on spark/spark2/thor/orin + /mesh/roster fingerprint agreement
- Stopping a specialist lane yields a visible error (404/503) for model=<lane>, never a 200 from another checkpoint or a cloud call.
  - instruction: acceptance: stop the lane container, request it, record status code + body
- Each named consumer has a documented opt-in path to a specialist lane by model id; none is switched implicitly.
  - instruction: docs grep for `EIDETIC_EMBED_MODEL` / `COHERENCE_EMBED_MODEL` guidance
- The after-state is shown by one acceptance transcript on the live mesh, not by unit tests alone (#108).
  - instruction: docs/evidence/<date>-accept-orin-embed.txt exists and is cited by the per-model docs
- The baseline numbers cited are the 2026-10-07 live readings.
  - instruction: same baseline section as c2
- The thresholds are measured on the physical AGX Orin under the shipped orin-embed render, not extrapolated from Spark or vendor numbers.
  - instruction: evidence header records box, shape, image digests
- The 5-point nDCG@10 threshold and the corpus are fixed before any candidate is scored.
  - instruction: commit corpus + threshold before the results commit
- The sidecar image pins exact transformers / sentence-transformers / torch versions and boots on the physical Orin, with the pins recorded in the per-model doc.
- The rendered compose publishes no host port for any specialist lane.

## Success signals

- On the AGX Orin: all chosen lanes co-resident with min available host memory >= 2 GiB under a load probe, zero container restarts over the acceptance run, and EmbeddingGemma 2 text/image/audio/video probes each pass a negative-control retrieval check (correct item ranked 1 of >= 3).
  - instruction: record in docs/evidence/<date>-accept-orin-embed.txt: free -m min, docker inspect RestartCount, per-modality probe output
- The 8B code specialist is chosen by a head-to-head on our own repos: it beats EmbeddingGemma 2 by >= 5 points nDCG@10 on NL->code and issue->source retrieval, or EmbeddingGemma 2 takes the code slot and the 8B slot goes to the next-best measured candidate.
  - instruction: commit the corpus + script + results table under docs/evidence/ and the per-model doc

## Scope / boundaries

- The plain embedder/reranker roles stay Qwen3-Embedding-0.6B / Qwen3-Reranker-0.6B on every box, Orin included: swapping them changes the vector space consumers pin and splits the mesh fingerprint pool.
- No embedding request ever falls back across checkpoints or to a cloud provider: an unavailable specialist lane fails visibly (404/503), never answers from a different vector space.
- Specialist lanes are reachable only through the gateway (bearer-gated when `GATEWAY_API_KEY` is set) — no host-published container port.

## Non-goals

- jina-code-embeddings is not a candidate: its licence is CC-BY-NC-4.0.
- Out of this Orin scope (stay on issue #291's other phases): the Gemini Embedding 2 cloud provider, LCO-Embedding-Omni-7B, the SigLIP 2 training-base track, and re-indexing eidetic/coherence onto a new space (consumers opt in by model id).

## Assumptions

- EmbeddingGemma 2 is served by a Python sidecar (Sentence-Transformers/transformers) following the realtime sidecar pattern — server module in the wheel, own Dockerfile, compose service, gateway route — because vLLM pooling support for EmbeddingGemma2Model is unverified.
- Every Orin embedding lane runs bf16 or W4A16 — `sm_87` has no NVFP4 W4A4 path — so an 8B embedder costs roughly 16 GiB resident.
- The EmbeddingGemma 2 sidecar can be built on the Orin's stack: it requires transformers 5.18.0.dev0, sentence-transformers >= 6.1.0 and torch 2.14 cu130; the Orin is L4T R39 and already runs a CUDA-13 vLLM nightly digest (sha256:7c5a10e9…) for its embed lane, so a CUDA-13 aarch64 image is plausible but unbuilt and untested on `sm_87`.

## Scope exploration

- `s1` — `orin live state (ssh orin: docker ps, free -m, ~/.lobes/.env)`: vllm-embed + vllm-rerank healthy 3 weeks; model-gear-vllm-associate Exited (1) 9 days ago; free -m available 46033 MiB, swap 0; .env still renders the orin-associate shape
  - seeds: `c2`
- `s2` — `lobes/gateway/_config.py + fleet docker-compose.yml (embed-deep)`: embed-deep is an opt-in second task=embed backend wired only when `EMBED_DEEP_BASE_URL` is set (`_config.py`:1014-1029), compose service vllm-embed-deep profile-gated (docker-compose.yml:495-518); `resolve_model` already routes N embed backends by model= (`_routing.py`:425-453). Hard-coded per-lane touch points: `FEASIBLE_ENV` (:136), `EMBED_MAX_ACTIVE` (:256), the self-named alias tuple (:785).
  - seeds: `c3`
- `s3` — `docs/colleague-stack.md 'Adding a role is effectively irreversible'`: lines 337-376: a different checkpoint is a catalog change, budget a profile/shape change; a new role only when nothing else expresses it — a code/multimodal embedder does not clear that bar; `tier_aliases` are generate-only with upward fallback, so not a tier either
  - seeds: `c3`
- `s4` — `consumers: eidetic-cli + coherence-cli`: eidetic/memory/embed.py:40 pins Qwen/Qwen3-Embedding-0.6B (override `EIDETIC_EMBED_MODEL` :175); coherence/meaning/embed.py:39-40 pins the same and its anchors are model-relative (:24, `refresh_meaning_vectors.py`)
  - seeds: `c4`
- `s5` — `lobes/gateway/_mesh_routing.py compute_role_placement`: a member whose embedder fingerprint disagrees is exposed only as embedder-<member> (:835-846); plain embedder stays the agreeing pool — a divergent Orin embedder would silently drop out of the plain pool
  - seeds: `c4`
- `s6` — `lobes/gateway/_routing.py order_backends`: returns 0 or 1 backend, 'No cross-backend failover, ever' (:570-576); unknown ids 404 `model_not_found` (:456-480); embed-deep has no upward fallback (`_config.py`:775-780) — issue #291 makes the same rule a requirement
  - seeds: `c5`
- `s7` — `lobes/roles.py RoleInfo / role_payload`: RoleInfo (:588-625) carries model/runtime/context/quant/mtp/responsibilities but no dimension, modality or MRL; dimension lives only in catalog.py (:106, `hf_overrides`). Contract tests: tests/`test_colleague_contract.py`, `test_gateway_capabilities.py`, `test_roles.py`
  - seeds: `c6`
- `s8` — `HF google/embeddinggemma-2 (API + model card)`: exists (created 2026-09-14), Apache-2.0, arch EmbeddingGemma2Model, 740M total (270M text + 170M vision + 300M audio), 768-d native, MRL 128/256/512, 8192 ctx, MTEB-code 78.68; library transformers/sentence-transformers — card never mentions vLLM
  - seeds: `c7`
- `s9` — `lobes/realtime/*_server.py + lobes/templates/fleet/Dockerfile.*`: `chatterbox_server.py`, `bluetts_server.py`, `listen_server`\*.py each ship a wheel module + Dockerfile (Dockerfile.chatterbox/.bluetts/.whisper-stt) + a docker-compose.audio.yml service the gateway fans out to via \*`_URL` — the precedent a non-vLLM embedding sidecar would copy
  - seeds: `c8`
- `s10` — `vLLM pooling docs + OpenAI /v1/embeddings`: stock OpenAI input is text/tokens only; vLLM adds a messages field with `image_url`/video parts and a /v2/embed Cohere-style route; no audio path found in vLLM pooling (docs.vllm.ai/en/latest/models/`pooling_models`/embed/)
  - seeds: `c9`
- `s11` — `issue #291 2026-10-07 comment`: requires EG2 vs Qwen comparison on code/repo retrieval before assigning a code-specialist default; Qwen3-VL pair stays available regardless of size
  - seeds: `c10`
- `s12` — `code-embedder licences (HF API)`: nomic-ai/nomic-embed-code apache-2.0; Qwen/Qwen3-VL-Reranker-8B apache-2.0 (exists); nvidia/Nemotron-3-Embed-8B-BF16 licence 'other' (OpenMDW-1.1 per issue)
  - seeds: `c10`
- `s13` — `HF jinaai/jina-code-embeddings-0.5b`: API cardData.license = cc-by-nc-4.0, despite its strong CoIR claim (73.94, jina's own)
  - seeds: `c11`
- `s14` — `lobes/profiles/builtin/orin.toml + docs/orin-profiles.md`: W4A4 NVFP4 needs Blackwell FP4 (orin.toml:125-131, orin-profiles.md:114-115); embed/rerank carry `TRITON_ATTN` + `enforce_eager` inherited from Thor, unretested on `sm_87` (orin-profiles.md ~119)
  - seeds: `c12`
- `s15` — `lobes/profiles/shapes.py + shape_render.py + tests/goldens`: shapes are pure TOML data (`load_builtin_shape` :306, `shape_env` :397, `overcommitted_groups` :360); new shape needs tests/goldens/shapes/orin-`embed__`{base,orin,spark,thor}.env via tests/goldens/regen.py
  - seeds: `c13`
- `s16` — `docs/orin-associate-deployment.md + 2026-09-13 evidence`: associate at util 0.70 / 1M leaves min available 1,925 MiB (:294-312) — no 8B embedder fits beside it; embed already restarts once when booted after associate (RestartCount 1)
  - seeds: `c13`
- `s17` — `mesh member lanes (docs/evidence/2026-10-06-accept-mesh-member-lanes.txt)`: member lanes embedder-orin / reranker-orin are live (:123-124) but are per-ROLE; whether the mesh roster announces non-role backends like embed-deep was not found in `_mesh_config.py`/`_mesh_roster.py`
  - seeds: `c14`
- `s18` — `issue #291 phases D/E + consumer env overrides`: Gemini is phase D (cloud), SigLIP 2 phase E (research); eidetic `EIDETIC_EMBED_MODEL` and coherence `COHERENCE_EMBED_MODEL` already let a consumer opt into a different lane
  - seeds: `c15`
- `s19` — `deferred questions`: c13 (associate's fate) and c10 (one winner vs several resident lanes) need the operator's decision; mesh reach for non-role backends is parked blocking
  - seeds: `q1` (question, resolved), `q2` (question, resolved)
- `s20` — `lobes/gateway/_mesh_routes.py announcement_from_capabilities + _mesh_wire.py`: the mesh Announcement is built ONLY from this box's own /capabilities payload, keyed by role (`_mesh_routes.py`:979-1003, wire RoleInfo per role lane `_mesh_wire.py`:90-113); /capabilities is built from lobes/roles.py ROLES, and embed-deep / multimodal-coder appear only in `_config.py` (:785, :1023) — so a self-named lane is never advertised and never mesh-reachable
  - seeds: `c20`, `c14`
- `s21` — `challenge pass / adjacent-systems lens: ../colleague/CLAUDE.md:147-157 + lobes CLAUDE.md associate section`: colleague ships an opt-in associate seat resolved by role name; `code_survey` runs on it when armed; lobes CLAUDE.md names the Orin as associate's only host
  - seeds: `c27`
- `s22` — `challenge pass / hidden-dependency lens: HF config_sentence_transformers.json + ssh orin (nv_tegra_release, docker inspect)`: EG2 pins a transformers DEV build + ST>=6.1.0 (multimodal dict ordering); Orin reports R39 and model-gear-vllm-embed runs vllm/vllm-openai@sha256:7c5a10e9…
  - seeds: `c28`
- `s23` — `challenge pass / failure-mode lens: google/embeddinggemma-2 model card lines 189-200`: card: in float16 the model returns NaN or silently degraded embeddings rather than raising; bfloat16 is the recommended default
  - seeds: `c29`
- `s24` — `challenge pass / data-flow lens: EG2 config_sentence_transformers.json prompts + card lines 118-151`: EG2 defines 20 named prompts with no default (`default_prompt_name` null); query 'task: code retrieval | query:' vs document 'title: … | text:' — an unprompted /v1/embeddings call would embed both sides the same way and understate retrieval quality
  - seeds: `c30`
- `s25` — `challenge pass / containment lens: ssh orin docker logs model-gear-vllm-associate + operator memory 2026-09-13`: associate's Exit(1) tail is an engine-launch failure ('generator didn't yield' / 'pickle data was truncated'); on this zero-swap board an uncapped process OOM-killed associate once before; embed already restarts once when booted after associate
  - seeds: `c31`
- `s26` — `challenge pass / reversibility lens: CLAUDE.md deployment shapes ('byte-for-byte restorable by re-running with the previous shape')`: shape switch is advertised as reversible; the new shape must keep that property — not yet exercised for an Orin shape that adds a sidecar
  - seeds: `c32`
- `s27` — `challenge pass / operations lens: lobes/cli/_commands/up.py:121-140`: up targets are keyed off lobes.roles.ROLES plus the gateway; embed-deep has no lobes up target today
  - seeds: `c33`
- `s28` — `challenge pass / security lens: lobes/templates/fleet/docker-compose.yml embed services + gateway auth`: existing embed/rerank lanes sit on the compose network behind the gateway; a sidecar copying the audio pattern must keep that posture
  - seeds: `c34`
- `s29` — `challenge pass / public-contract lens: tests/test_colleague_contract.py:402 + ../colleague/colleague/config_defaults.py:271`: set(contract) == set(ROLES) is asserted; `announcement_from_capabilities` iterates top-level entries; colleague reads /capabilities by role name
  - seeds: `q4` (question, resolved), `c20`
- `s30` — `challenge pass / concurrency lens: lobes/gateway/_config.py:256 EMBED_MAX_ACTIVE`: max-active is per named embed backend and hard-coded; each specialist lane needs its own key — already covered by c3's per-lane touch points; no new claim
- `s31` — `challenge pass / residual`: examined: adjacent systems, hidden deps, failure modes, data flow, containment, reversibility, operations, security, public contract, concurrency. NOT examined: Nemotron/Qwen3-VL actual `sm_87` boot (parked v2), gateway messages passthrough (parked v3), Orin thermal/power mode under sustained load, embeddings-cli backend ids
- `s32` — `challenge pass / public-contract lens: ../colleague/colleague/lobes.py:56-61`: colleague treats /capabilities as a superset and parses only role keys via `_parse_role` — extra top-level lane keys are tolerated
  - seeds: `c36`

## Decisions

- The Orin drops the associate lane and becomes the mesh's embedding specialist; lanes are replaceable as needs change.
- The Orin hosts several embedding lanes resident at once: EmbeddingGemma 2 (multimodal) + an 8B code/retrieval specialist, optionally a fine-tunable embedding base.
- Qwen3-VL-Reranker-8B is added only if the measured Orin memory budget fits it alongside the embedders.
- The fine-tunable embedding base is EmbeddingGemma 2 itself (small, Apache-2.0, cheapest path via Sentence-Transformers); fine-tunes are produced out of tree and served under a new distinct model identity.
- Each specialist lane is a distinctly named top-level /capabilities key, auto-reflected into the mesh announcement; no nested lanes section.

## Hard questions

- Serve several specialist lanes at once (EG2 multimodal + an 8B code/retrieval specialist resident together), or keep only the single winner of the head-to-head resident and the others as switchable candidates? (resolved: Several lanes resident at once: EmbeddingGemma 2 plus an 8B code/retrieval specialist, possibly plus a fine-tunable embedding model (operator, 2026-10-07).)
- Does the Orin also take a matched rich reranker (Qwen3-VL-Reranker-8B, Apache-2.0), or stay on the 0.6B reranker only? (resolved: Add Qwen3-VL-Reranker-8B if the measured Orin memory budget fits it beside the embedders; otherwise stay on the 0.6B reranker (operator, 2026-10-07).)
- Does the Orin give up the associate lane entirely to become the embedding specialist (it is already Exited 9 days), or keep a shrunk associate (e.g. 128K at util ~0.56) beside the embedders? (resolved: Associate is dropped from the Orin: a wide range of embedding models is worth more, and models are replaceable as needs change (operator, 2026-10-07).)
- The /capabilities payload is contractually keyed EXACTLY by ROLES (tests/`test_colleague_contract.py`:402) and colleague resolves seats by role name — do specialist lanes go in as extra top-level keys (contract change) or in a new nested section (e.g. 'lanes') that the mesh announcement builder must also read? (resolved: Top-level keys: each specialist lane is its own distinctly named /capabilities key (they are not the same lane), so the existing mesh announcement reflects them automatically (operator, 2026-10-07).)

## Open parks

- [unknown_nonblocking] Whether the pinned vLLM nightly (`sm_87`) serves EmbeddingGemma2Model, Qwen3-VL-Embedding-8B, or Nemotron-3-Embed-8B under --runner pooling — spike before choosing vLLM vs sidecar per model.
- [unknown_nonblocking] Whether the lobes gateway passes a vLLM-style messages body through /v1/embeddings unmodified, or validates/strips it.
- [unknown_nonblocking] /v1/embeddings has NO request-body cap (server.py `_post_body_limit` is render-scoped only), so a base64 video/audio embed request is buffered whole in gateway memory — does the multimodal lane need its own cap?
- [follow_up] Why model-gear-vllm-associate on the Orin exited (1) ~9 days ago — logs not read; independent of this feature but determines whether associate is worth keeping.

## Resolved vagueness

- [unknown_blocking] Whether the mesh roster announces/auto-wires non-role embed backends (embed-deep-style self-named lanes) at all — if not, cross-box reach needs mesh work or a role-level answer. — resolved: Answered by reading the code: non-role backends are NOT announced (mesh announcement = own /capabilities, role-keyed). Reach needs advertising specialist lanes in /capabilities — captured as c20.
