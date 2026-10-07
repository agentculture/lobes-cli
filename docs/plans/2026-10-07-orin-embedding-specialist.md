# Build Plan — orin-embedding-specialist

slug: `orin-embedding-specialist` · status: `exported` · from frame: `orin-embedding-specialist`

> The Jetson AGX Orin is the mesh's embedding specialist: it hosts several distinctly-addressed embedding lanes — a compact multimodal one (EmbeddingGemma 2: text, code, image, video, audio in one 768-d space) and a code/retrieval specialist picked by a measured head-to-head — each advertising its concrete vector-space identity, reachable from every mesh member, with no cross-model fallback and the 0.6B default embedder untouched.

## Tasks

### t1 — Spike: which Orin lanes vLLM pooling can serve

- instruction: Read-only against the deployment: run throwaway containers with --rm, a --memory cap, and `HF_HUB_DISABLE_XET`=1 (the Orin has zero swap; an uncapped download OOM-killed associate on 2026-09-13). Do not touch ~/.lobes. Associate is already Exited, so memory is free. Resolves parked v2.
- acceptance:
  - docs/evidence/<date>-spike-orin-embed-vllm.txt records, for google/embeddinggemma-2, Qwen/Qwen3-VL-Embedding-8B, Qwen/Qwen3-VL-Reranker-8B and nvidia/Nemotron-3-Embed-8B-BF16, whether the Orin's pinned vLLM digest boots it under --runner pooling (bf16), with the exact error or a passing /v1/embeddings (or /v1/rerank) call
  - each model ends the spike labelled vllm or sidecar, which later tasks consume

### t2 — Retire associate from the Orin: docs plus a colleague issue

- instruction: Use the communicate skill for the cross-repo issue. Edit prose only. The docs land before orin-embed is applied (h20). Note what associate's Exit(1) log showed (parked v4) without guessing a cause.
- covers: c27, h20
- acceptance:
  - an issue on agentculture/colleague says the associate seat has no host once the Orin renders orin-embed, and links this plan
  - the CLAUDE.md associate section and docs/orin-associate-deployment.md say associate is DORMANT/unhosted (like muse), with orin-associate kept in-tree as the rollback shape

### t3 — Specialist-lane registry with name-collision rules

- instruction: Each lane gets its own distinct name (operator decision c35); use a short model slug such as gemma2-embed, never anything starting with embedder- or reranker-. Pure data plus a validator; no I/O.
- covers: c3, h3, h27
- acceptance:
  - new lobes/`embed_lanes.py` declares each specialist lane (name, catalog id, engine, base-URL env key, task embed/score, modalities, native dim, MRL dims, normalization) as a frozen dataclass tuple; lobes.roles.ROLES is unchanged
  - tests/`test_embed_lanes.py` refuses a lane name equal to any role, any tier alias, or matching the {role}-{member} member-lane pattern
- obligation: `o1` (criterion 2) [lane registry name validation] a lane name equal to a role, a tier alias, or matching {role}-{member} is refused at import/test time

### t4 — Catalog: vector-space identity fields and the new embedding entries

- instruction: `role_hint` must be 'candidate' so `_catalog_by_role_hint` never lets one hijack the 0.6B embedder (catalog.py:440-443). No jina-code entry (CC-BY-NC, c11).
- covers: c6
- acceptance:
  - SupportedModel gains modalities, `mrl_dims` and normalization fields (defaulted, so existing entries render byte-identical); ENGINES gains a sentence-transformers engine
  - catalog entries exist for google/embeddinggemma-2 (Apache-2.0, 768, MRL 128/256/512, text/code/image/video/audio), Qwen3-VL-Embedding-8B, Qwen3-VL-Reranker-8B, Nemotron-3-Embed-8B-BF16 (license recorded as OpenMDW-1.1) and nomic-embed-code; none has `role_hint` embedding/reranker; status is 'configured' until measured

### t5 — EmbeddingGemma 2 sidecar server module

- instruction: Mirror the lobes/realtime/\*`_server.py` pattern: a thin pragma-no-cover app shell over pure, tested functions. Modular load (text-only vs full) is selected by env. Health endpoint for the compose healthcheck.
- covers: c29, h21, c30, h22, c9
- acceptance:
  - new lobes/`embed_sidecar`/server.py serves /v1/embeddings: text input, a messages body with image/audio/video parts, a `prompt_name` field, and a dimensions field (MRL truncate, then re-normalize); unsupported dims are refused
  - responses echo model, applied prompt, dimensions and loaded modalities; a NaN/inf/all-zero vector is an error, not a 200; float16 is refused at load (tests/`test_embed_sidecar.py` with an injected fake encoder)
  - no code path turns a non-text part into text
- obligation: `o2` (criterion 2) [sidecar /v1/embeddings response] a NaN, inf or all-zero vector returns an error, never 200; loading in float16 is refused
- obligation: `o3` (criterion 1) [sidecar dimensions parameter] MRL-truncated vectors come back re-normalized to unit length; a dimension outside 128/256/512/768 is refused
- obligation: `o4` (criterion 3) [sidecar multimodal input] image/audio/video parts are encoded by their own encoder, never converted to text

### t6 — Gateway: wire specialist lanes from the registry, with no fallback

- instruction: Generalise the embed-deep pattern (`_config.py`:1014-1029, 772-812) by iterating the registry rather than hard-coding names. Leave embed-deep's own behaviour byte-identical.
- depends on: t3, t4
- covers: c5, h5
- acceptance:
  - lobes/gateway/`_config.py` wires one optional backend per registry lane behind its own \*`_BASE_URL`, with its own FEASIBLE and `MAX_ACTIVE` keys; model=<lane> and the raw checkpoint id both resolve
  - tests/`test_gateway_embed_lanes.py`: an unwired or stopped lane answers 404/503 and never another lane; a multimodal messages body reaches the upstream byte-identical (parked v3)
- obligation: `o5` (criterion 2) [gateway model routing] an unwired or stopped lane answers 404/503 and is never served by another lane, checkpoint or cloud provider
- obligation: `o6` (criterion 2) [gateway → lane request body] a multimodal messages body reaches the upstream byte-identical

### t7 — Capabilities: each lane is a top-level /capabilities key; widen the role contract

- instruction: Touch lobes/gateway/server.py's `capabilities_payload` and the contract tests only. lobes capabilities (CLI) renders lanes in a separate block.
- depends on: t3, t4
- covers: c36, h6
- acceptance:
  - `capabilities_payload` adds one top-level key per wired lane carrying model, runtime, task, path, native dim, MRL dims, modalities, normalization, `tested_on`, ready/feasible and a fingerprint; the values match the catalog (tests/`test_gateway_capabilities.py`)
  - tests/`test_colleague_contract.py`:402/421 become 'every ROLES key present' plus a lane-key check; a test proves colleague's superset parser still resolves every seat with lane keys present
- obligation: `o7` (criterion 2) [/capabilities ↔ colleague parser] colleague resolves every seat from a payload that also carries lane keys

### t8 — Mesh: announce lanes, auto-wire them, forward by name

- instruction: Files: lobes/gateway/`_mesh_routes.py` and `_mesh_routing.py`, plus tests/`test_mesh_lanes.py`. Lanes are per-lane, never pooled across different checkpoints.
- depends on: t6, t7
- covers: c20, h13
- acceptance:
  - `announcement_from_capabilities` includes lane keys (test); a member that does not host a lane auto-wires it from a verified peer with no per-lane env and forwards model=<lane> in one hop with X-Lobes-Mesh-Member
  - a peer on an older build ignores unknown lane keys without failing verification of its roles (decode test)
- obligation: `o8` (criterion 1) [mesh announcement and forward] a non-hosting member forwards model=<lane> to the hosting peer in one hop with X-Lobes-Mesh-Member, with no per-lane env typed
- obligation: `o9` (criterion 2) [mixed-version mesh decode] a peer that does not know lane keys still verifies the announcer's roles

### t9 — CLI: start, stop and probe each lane individually

- instruction: Extend TARGETS in lobes/cli/`_commands`/up.py from the registry rather than ROLES.
- depends on: t3
- covers: c33, h25
- acceptance:
  - lobes up <lane> (dry-run by default, --apply to commit, always --no-deps) starts just that lane; lobes status lists lanes; lobes assess <lane> runs an embed or rerank probe with a negative control
  - tests/`test_cli_up.py`-style tests prove a lane target touches no other service
- obligation: `o10` (criterion 2) [lobes up <lane>] starts or stops only that lane's service (no dependency walk), dry-run by default

### t10 — Code-retrieval head-to-head harness and corpus

- instruction: HTTP-only against the gateway: it scores whatever model ids it is given. Each candidate gets its card's recommended prompts (EG2 CodeRetrieval/Document; the Qwen instruct format). The script has no network or mesh dependency of its own beyond the target URL.
- covers: c10, h9, h18
- acceptance:
  - scripts/`embed_h2h`/ holds a labelled corpus from our own repos (NL→code and issue/stacktrace→source, lobes-cli + culture at a pinned SHA), an nDCG@10 scorer, and per-model query/document prompts
  - the corpus and the 5-point threshold are committed in a commit before any results commit (h18)

### t11 — Fine-tuned EmbeddingGemma 2 lane mechanism

- instruction: Adapter/fine-tune production stays out of tree (Sentence-Transformers, operator-run). No fine-tune ships in this PR; only the mechanism.
- depends on: t5, t3
- covers: c19, h12
- acceptance:
  - the registry and sidecar accept a fine-tune lane pointing at a local checkpoint path under its own name and identity; its presence leaves the base lane's vectors identical (cosine 1.0 on a fixed probe set, offline test with fake encoders)
- obligation: `o11` (criterion 1) [base EmbeddingGemma 2 lane] adding a fine-tune lane leaves the base lane's vectors identical (cosine 1.0)

### t12 — Spike: build and boot the EmbeddingGemma 2 Sentence-Transformers image on the Orin

- instruction: Base the image on the CUDA-13 aarch64 stack the Orin already runs (L4T R39). Use uv pip install --system, not pip. Use throwaway --rm containers with a --memory cap and `HF_HUB_DISABLE_XET`=1. Skip only if t1 shows vLLM serves EmbeddingGemma2Model for every declared modality. Proves the c28 assumption.
- depends on: t1
- acceptance:
  - an aarch64 CUDA-13 image with pinned transformers (5.18.0.dev0 or the first release supporting EmbeddingGemma2Model), sentence-transformers>=6.1.0 and torch builds and runs on the Orin with bf16 (torch.cuda.`is_bf16_supported`() true on `sm_87`)
  - text, image, audio and video encode calls each return a finite 768-d vector; the pins and measured resident memory are recorded in the evidence file

### t13 — Compose overlay and Dockerfile for the embedding lanes

- instruction: Leave the base fleet docker-compose.yml untouched. Boot order is healthcheck-gated without `depends_on` edges on the gateway (compose up walks `depends_on`, issue #222).
- depends on: t12
- covers: c34, h26, c31
- acceptance:
  - new lobes/templates/fleet/docker-compose.embed.yml declares each specialist lane service (vLLM or sidecar, per t1) with a healthcheck, a container memory cap (`mem_limit`) and no ports: mapping; Dockerfile.embed-st builds the sidecar with the pins from the image spike
  - a template test asserts that no specialist service publishes a host port and that every one carries `mem_limit`
- obligation: `o12` (criterion 2) [rendered compose] no specialist service publishes a host port; every one carries `mem_limit` and a healthcheck

### t14 — orin-embed built-in shape, render knobs and goldens

- instruction: Follow orin-associate.toml's structure; the shape adds docker-compose.embed.yml to the rendered compose file set. orin-associate.toml stays in-tree, untouched.
- depends on: t3, t4, t13
- covers: c13, h10
- acceptance:
  - lobes/profiles/`builtin_shapes`/orin-embed.toml hosts embedder, reranker and the specialist lanes, does not host associate, and declares per-lane budgets (DECLARED until the deploy task measures them)
  - tests/goldens/shapes/orin-`embed__`{base,orin,spark,thor}.env are generated by tests/goldens/regen.py; every pre-existing golden is byte-unchanged; tests/`test_orin_embed_shape.py` passes
- obligation: `o13` (criterion 2) [pre-existing shape goldens] every pre-existing golden .env is byte-unchanged

### t15 — Deploy orin-embed on the Orin and measure every lane

- instruction: Use the lobes-deploy skill and lobes-compose.sh, never compose inside lobes/templates. Back up ~/.lobes first. Run the PR's TestPyPI dev wheel on the Orin gateway (allow for propagation delay).
- depends on: t6, t7, t9, t14
- covers: c7, h7, h23, c25
- acceptance:
  - docs/evidence/<date>-measure-orin-embed.txt records: EG2 full and text-only resident memory and p50 latency; each 8B lane's resident memory; the free -m minimum under a load probe; per-container `mem_limit`; RestartCount; a dmesg OOM grep
  - the Qwen3-VL-Reranker-8B include/exclude decision follows c18 (at least 2 GiB available with it loaded); the shape's DECLARED budgets are replaced by measured values

### t16 — Run the code head-to-head on the Orin and pick the 8B specialist

- instruction: Use the t10 harness against the Orin gateway. Never change the threshold after scoring.
- depends on: t10, t15
- covers: c26
- acceptance:
  - a results table (every candidate on identical queries) is committed under docs/evidence/, naming the chosen 8B code lane and applying the decision rule
  - the shape's 8B lane is set to the winner (or the next-best candidate, per c26)

### t17 — Exercise the rollback to orin-associate

- instruction: Do this before the acceptance run. The claim covers the rendered deployment files; associate need not serve.
- depends on: t15
- covers: c32, h24
- acceptance:
  - evidence shows lobes init --shape orin-associate --profile orin --apply --force diffing clean against the pre-change backup, then orin-embed re-applied

### t18 — Mesh-wide acceptance transcript

- instruction: The header records box, shape, image digests and wheel version. Spark, spark2 and thor must run a gateway that includes the mesh task (t8).
- depends on: t8, t16, t17
- covers: c1, h1, c2, h2, c4, h4, c5, h5, c14, h11, c23, h15, c24, h16, h17, h8
- acceptance:
  - docs/evidence/<date>-accept-orin-embed.txt holds the 2026-10-07 baseline and, from spark, spark2 and thor: GET /capabilities listing each Orin lane, plus model=<lane> calls answered with X-Lobes-Mesh-Member: orin; plain embedder is still Qwen/Qwen3-Embedding-0.6B and orin stays in the agreeing embedder pool
  - it also shows a stopped lane answering 404/503 (then restarted), per-modality EG2 probes ranking the correct item 1st of at least 3, RestartCount 0 and no OOM lines
- obligation: `o14` (criterion 1) [plain embedder role mesh-wide] model=embedder still answers with Qwen/Qwen3-Embedding-0.6B on every box, and orin stays in the agreeing pool

### t19 — Docs: per-model docs, the Orin deployment doc and consumer opt-in

- instruction: Use the #108 honesty vocabulary (MEASURED / DECLARED / UNVALIDATED) exactly.
- depends on: t2, t16
- covers: c22, h14
- acceptance:
  - docs/embeddinggemma-2.md, a doc for the chosen 8B lane(s) and docs/orin-embed-deployment.md cite the evidence files; a consumer section shows `EIDETIC_EMBED_MODEL` / `COHERENCE_EMBED_MODEL` opt-in by lane id and states that no consumer is switched implicitly
  - CLAUDE.md, the lobes explain catalog and docs/colleague-stack.md's lane description are updated; markdownlint passes

### t20 — PR obligations: bump, lint, rubric, lock capture

- instruction: Pre-empt Sonar S3776/S1192 in the new modules before pushing.
- depends on: t11, t18, t19
- acceptance:
  - version bumped (minor) via the version-bump skill with a CHANGELOG entry; uv.lock re-pinned
  - uv run pytest -n auto, black --check, isort --check-only, flake8, bandit, afi cli doctor . --strict and markdownlint all pass; the doc-test-alignment skill has run; secrets-scan is clean
  - the Orin deployment lock is captured into deployments/jetson-agx-`orin__orin`-embed/ with a VARIATION.md citing the acceptance transcript

### t21 — Run /validate-delivery

- instruction: Never suppress a partial outcome. Check every confirmed plan obligation (devague plan oblige --list).
- depends on: t20
- acceptance:
  - the validate-delivery skill ran the plan's behavioral tests and filed obligations as evidence or behavioral deltas via the devague CLI, failing or partial outcomes included

### t22 — Run /summarize-delivery

- instruction: Runs on complete, partial and failed runs alike.
- depends on: t21
- acceptance:
  - docs/deliveries/<date>-orin-embedding-specialist.md records planned vs actual, deviations, evidence-backed delivery claims and remaining work

### t23 — Open the PR and drive review to merge

- instruction: Network-bound gh/agex calls need the sandbox disabled. Never merge with unaddressed comments.
- depends on: t22
- acceptance:
  - a PR opened via the cicd skill references #291 and the spec, plan and delivery docs; every review thread is answered and resolved; the Sonar gate passes (`SONAR_PROJECT_KEY`=`agentculture_model`-gear); CI is green

## Risks

- [unknown_nonblocking] vLLM pooling may not serve some candidates on the Orin's `sm_87` digest (frame park v2) — the t1 spike decides vllm vs sidecar per model (task t1)
- [unknown_nonblocking] EmbeddingGemma 2 depends on a transformers DEV build (5.18.0.dev0); a moving pin could break the sidecar image — pin by version or commit, record it (task t12)
- [unknown_nonblocking] the 8B embedder + 8B reranker + EG2 may not fit with >= 2 GiB free on the zero-swap Orin; the c18 rule drops the reranker first (task t15)
- [unknown_nonblocking] /v1/embeddings has no request-body cap (frame park v5); base64 video is buffered whole in gateway memory — measure in t15, cap if needed (task t15)
- [unknown_nonblocking] spark/spark2/thor gateways must be re-imaged with the mesh-lane build before acceptance (stale-advert trap from the 2026-09-11 run) (task t18)
