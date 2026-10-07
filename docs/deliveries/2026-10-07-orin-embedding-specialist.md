# Delivery Summary — orin-embedding-specialist

plan: `orin-embedding-specialist` · run: `complete` (with deviations d1/d2 and one failing obligation, o14) · date: `2026-10-07`
baseline: `devague summary skeleton`

## Intent

Make the Jetson AGX Orin the mesh's embedding specialist (#291). Associate is dropped, and the Orin hosts distinctly named specialist embedding lanes. Each lane advertises its vector-space identity, is reachable by name from every mesh member, and never falls back to another model. The plan as confirmed asked for EmbeddingGemma 2 plus a measured 8B code/retrieval specialist. The operator narrowed it mid-run (`d1`) to EmbeddingGemma 2 as the only standard lane.

## Planned Work

Quoted verbatim from the `devague summary` skeleton:

- `t1` — Spike: which Orin lanes vLLM pooling can serve
- `t2` — Retire associate from the Orin: docs plus a colleague issue
- `t3` — Specialist-lane registry with name-collision rules
- `t4` — Catalog: vector-space identity fields and the new embedding entries
- `t5` — EmbeddingGemma 2 sidecar server module
- `t6` — Gateway: wire specialist lanes from the registry, with no fallback
- `t7` — Capabilities: each lane is a top-level /capabilities key; widen the role contract
- `t8` — Mesh: announce lanes, auto-wire them, forward by name
- `t9` — CLI: start, stop and probe each lane individually
- `t10` — Code-retrieval head-to-head harness and corpus
- `t11` — Fine-tuned EmbeddingGemma 2 lane mechanism
- `t12` — Spike: build and boot the EmbeddingGemma 2 Sentence-Transformers image on the Orin
- `t13` — Compose overlay and Dockerfile for the embedding lanes
- `t14` — orin-embed built-in shape, render knobs and goldens
- `t15` — Deploy orin-embed on the Orin and measure every lane
- `t16` — Run the code head-to-head on the Orin and pick the 8B specialist
- `t17` — Exercise the rollback to orin-associate
- `t18` — Mesh-wide acceptance transcript
- `t19` — Docs: per-model docs, the Orin deployment doc and consumer opt-in
- `t20` — PR obligations: bump, lint, rubric, lock capture
- `t21` — Run /validate-delivery
- `t22` — Run /summarize-delivery
- `t23` — Open the PR and drive review to merge

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | `docs/evidence/2026-10-07-spike-orin-embed-vllm-8b.txt`: Nemotron and nomic serve at util 0.35; Qwen3-VL-Embedding serves only at 0.52; the Qwen3-VL reranker is refused; vLLM refuses EG2 |
| `t2` | delivered | the associate DORMANT notes in CLAUDE.md and `docs/orin-associate-deployment.md` (`7fafa2e`); agentculture/colleague#500 |
| `t3` | delivered | `lobes/embed_lanes.py` with its name-collision validator |
| `t4` | delivered | catalog identity fields plus 5 entries; 4 measured models now `load-tested` |
| `t5` | delivered | `lobes/embed_sidecar/server.py`; the live HTTP-shell bug was fixed in `a4aa2d0`, and `4521ec8` added the selective encoder load |
| `t6` | delivered | lane wiring in `lobes/gateway/_config.py`, no cross-lane fallback |
| `t7` | delivered | top-level `/capabilities` lane keys; the widened colleague contract test |
| `t8` | delivered | mesh lane forwarding; it also fixed a bug where `_peer_only_forward` could send an unknown lane id to a peer's `primary` |
| `t9` | delivered | `lobes up/status/assess <lane>` |
| `t10` | delivered | `scripts/embed_h2h/` harness, corpus and threshold, committed before any result (`c5ebec0`) |
| `t11` | delivered | `EMBED_FINETUNE_LANES` with `local:<name>` identity |
| `t12` | delivered | `docs/evidence/2026-10-07-spike-embeddinggemma2-sidecar-orin.txt` |
| `t13` | delivered | `docker-compose.embed.yml` and `Dockerfile.embed-st`; the `code` default-modality bug was fixed before deploy |
| `t14` | delivered | `orin-embed` shape and goldens |
| `t15` | partial | live on the Orin with **gemma2-embed only** (`d1`); measured budgets in the shape (`01a4baf`); Nemotron measured but opt-in |
| `t16` | partial | head-to-head ran for all 4 candidates (`docs/evidence/2026-10-07-h2h-code-retrieval-orin.txt`); the c26 rule gives EG2 the code slot; the broader evaluation went to #296 (`d2`) |
| `t17` | delivered | rollback rehearsal; re-render alone is incomplete and the backup restore is the rollback |
| `t18` | delivered | `docs/evidence/2026-10-07-accept-orin-embed.txt`; obligation o14 FAILS on a pre-existing rule (#297) |
| `t19` | delivered | 6 docs plus CLAUDE.md, `colleague-stack.md` and `lobes explain lanes` |
| `t20` | delivered | 0.84.0 bump, CHANGELOG, lint, bandit, `afi --strict`, secrets scan, Orin lock captured; `doc-test-alignment` is a stub and checked nothing |
| `t21` | delivered | 17 evidence records (16 pass, 1 fail), 5 deltas, 2 lapses, all confirmed |
| `t22` | delivered | this file |
| `t23` | blocked | not yet done: PR #295 is still a draft at the time of writing; marking it ready and review follow |

## Mid-work Decisions

- `d1` — orin-embed ships with EmbeddingGemma 2 (gemma2-embed) as the ONLY standard specialist lane beside the 0.6B pair; Nemotron-3-Embed-8B is NOT standard — it stays a declared opt-in candidate (catalog + overlay service + compose profile), not in the shape's hosted lanes — Operator decision 2026-10-07 after the t16 head-to-head: Nemotron measurably beat EG2 only on 24 issue->source queries (+8.4 nDCG@10, no CI) at ~4.4x the memory (20.6 vs 4.7 GiB); NL->code was saturated; 'bigger is not always better'. Supersedes decision c17's 'EG2 + an 8B specialist resident together'.
- `d2` — the broader evaluation (prose/docs/articles/stories/math, real image/video/audio retrieval with human-caption labels, MRL quality, cross-model agreement, and a redone code test with git-history labels + bootstrap CIs) moves to a follow-up issue instead of this run — Operator decision 2026-10-07: everything except code was only smoke-tested (encode paths + 2-3 toy items); the t16 code test itself has label-leakage/saturation/small-n weaknesses. A proper evaluation round is its own piece of work under #291.

- **Draft PR opened early (operator, 2026-10-07)** so CI publishes a TestPyPI dev wheel for the live boxes; it is not ready for review until t23.
- **Nemotron chosen for the 8B slot, then made non-standard.** The operator first picked Nemotron as the 8B slot after the c26 rule fired, then narrowed the shape to EG2 only (`d1`).
- **ASSOCIATE_FEASIBLE=false hand-added on the live Orin.** `.env` is merge-only, so a stale `ASSOCIATE_BASE_URL` from the old shape kept associate advertised as feasible.
- **All four mesh gateways re-imaged to 0.84.0.dev619** with `up -d --no-deps gateway` over each container's own compose chain, so lanes forward across the mesh.

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|------------------------|-----------------|
| `t15` (`d1`) | Operator decision 2026-10-07 after the t16 head-to-head: Nemotron measurably beat EG2 only on 24 issue->source queries (+8.4 nDCG@10, no CI) at ~4.4x the memory (20.6 vs 4.7 GiB); NL->code was saturated; 'bigger is not always better'. Supersedes decision c17's 'EG2 + an 8B specialist resident together'. | `acceptable` |
| `t16` (`d2`) | Operator decision 2026-10-07: everything except code was only smoke-tested (encode paths + 2-3 toy items); the t16 code test itself has label-leakage/saturation/small-n weaknesses. A proper evaluation round is its own piece of work under #291. | `needs-follow-up` |

| `t18` | obligation o14 ("orin stays in the agreeing pool") cannot hold: identical pooling-gear fingerprints never pool (`unknown` never pools, #233/#252) — pre-existing, filed #297 | `needs-follow-up` |
| `t20` | `doc-test-alignment` is a stub (exits not-implemented); docs/test alignment was checked by hand only | `acceptable` |
| `t23` | PR still draft at summary time | `needs-follow-up` |

## Evidence

- tests: full suite `uv run pytest -n auto` — 6487 passed, 17 skipped at `ab4eae8`; obligation tests (o1–o13) re-run at `ab4eae8` — all pass (see `devague evidence --list`, e1–e16)
- live: `docs/evidence/2026-10-07-accept-orin-embed.txt` (mesh acceptance), `...-spike-embeddinggemma2-sidecar-orin.txt`, `...-spike-orin-embed-vllm-8b.txt`, `...-h2h-code-retrieval-orin.txt`, `...-rollback-orin-embed-to-orin-associate.txt`
- lint: black / isort / flake8 clean; bandit no findings; `uv run afi cli doctor . --strict` exit 0; `markdownlint-cli2` clean on docs/CLAUDE.md/deployments; `scan_deployment_secrets.py` clean
- commits: `d1017b0..ab4eae8` (44 commits) on `spec/orin-embedding-specialist`
- PRs / issues: PR #295 (draft) · #291 · #296 (evaluation round) · #297 (pooling finding) · agentculture/colleague#500

## Delivery Claims

| Claim | Confidence | Evidence |
|-------|------------|----------|
| The Orin serves EmbeddingGemma 2 as `gemma2-embed` (text/image/audio/video → one 768-d space, bf16, prompt names, MRL 128–768) | high | `docs/evidence/2026-10-07-accept-orin-embed.txt` §1–§2 · e5 · `tests/test_embed_sidecar.py` |
| Every mesh member reaches it by lane name and raw id in one hop (`X-Lobes-Mesh-Member: orin`), no per-lane env | high | e11 (live, spark/spark2/thor) · e10 `tests/test_mesh_lanes.py::test_non_hosting_member_forwards_lane_in_one_hop` |
| `/capabilities` advertises each lane's vector-space identity as a top-level key; colleague still resolves every seat | medium | e9 is fidelity-only (vendored parser, not the real client) · `tests/test_gateway_capabilities.py` |
| No cross-lane / cross-checkpoint / cloud fallback; a stopped lane answers 503 | high | e6 + e7 (live §3) |
| Lanes are memory-capped, health-checked, publish no host port; zero OOM kills since deploy | high | e15 · acceptance §5 |
| EG2 is "good" at images/audio/video | unverified | only synthetic ordering probes; evaluation is #296 |
| EG2 is the best code embedder for the mesh | low | lapse `l2` (n-below-claim: LLM-written labels, saturated NL→code, n=24 issue queries, no CI) caps it |
| The plain 0.6B `embedder` is unchanged and pooled mesh-wide | low — **FAILING** for the pool | e17 (fail): 200 on spark2/thor, 404 on the Spark — pre-existing rule, #297 |
| Rollback to orin-associate is one command | low | t17 evidence: a re-render alone is incomplete; the backup restore is the rollback |
| Fine-tune lanes keep the base vector space intact | medium | e14 (fake encoders only; no real fine-tune exists yet) |

Lapse ledger evidence:

| Lapse | Code | What |
|-------|------|------|
| `l1` | `grader-unverified` | committed with one failing test (the `&&` chain read `tail`'s exit code); caught and amended before push |
| `l2` | `n-below-claim` | the t16 code head-to-head fed the c26 decision despite LLM-written labels, NL→code saturation and n=24 with no CI |

## Remaining Work / Follow-up

- `t23` — mark PR #295 ready, wait for reviewers, answer every thread, Sonar gate (`SONAR_PROJECT_KEY=agentculture_model-gear`), merge.
- #296 — the real evaluation round (prose/docs/articles/stories/math, real image/video/audio retrieval, MRL quality, cross-model agreement, redone code test) — decides whether any 8B lane (Nemotron) becomes standard.
- #297 — identical pooling-gear fingerprints never pool on a non-hosting box (o14).
- Republish: the boxes run `0.84.0.dev619` from TestPyPI; after merge, re-pin every gateway (and the Orin's sidecar build) to the released `0.84.0` and drop the TestPyPI extra index.
- The Orin's associate engine crash (`generator didn't yield` / `pickle data was truncated`) — never diagnosed; irrelevant while associate is dormant.
- Sidecar real-adapter paths (`SentenceTransformerEncoder`, `build_app`) stay `pragma: no cover`; only the FastAPI request shell has a regression test (`importorskip`).
- `doc-test-alignment` skill is a stub.
