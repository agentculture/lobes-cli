# Build Plan — model-gear now serves an embedding gear (Qwen3-Embedding-0.6B, 1024-dim) and a reranker gear (Qwen3-Reranker-0.6B) as always-warm fleet backends co-resident with the Qwen3.6-27B primary on the shared DGX Spark, and every consumer reaches all three through one OpenAI-compatible gateway routed by the request model field.

slug: `model-gear-now-serves-an-embedding-gear-qwen3-embe-2` · status: `exported` · from frame: `model-gear-now-serves-an-embedding-gear-qwen3-embe-2`

> model-gear now serves an embedding gear (Qwen3-Embedding-0.6B, 1024-dim) and a reranker gear (Qwen3-Reranker-0.6B) as always-warm fleet backends co-resident with the Qwen3.6-27B primary on the shared DGX Spark, and every consumer reaches all three through one OpenAI-compatible gateway routed by the request model field.

## Tasks

### t1 — Extend the catalog model: add task (generate|embed|score), dimension, and hf_overrides fields to SupportedModel; add Qwen/Qwen3-Embedding-0.6B (task=embed, dimension=1024) and Qwen/Qwen3-Reranker-0.6B (task=score) catalog entries.

- covers: c10
- acceptance:
  - SupportedModel gains task (default 'generate'), dimension (int|None, default None), hf_overrides (str, default '') fields; the two 0.6B gears appear in SUPPORTED_MODELS with correct task/dimension/hf_overrides; every existing chat entry keeps task='generate' with dimension=None; as_dicts() includes the new fields; black/isort/flake8 clean. File: model_gear/catalog.py.

### t2 — Catalog-consistency tests for the new fields: a task=embed model must have a non-null dimension and a docs/ file; a task=score model must carry the Qwen3ForSequenceClassification hf_overrides; generate models keep dimension=None.

- depends on: t1
- covers: h8
- acceptance:
  - tests/test_catalog.py asserts: every embed model has dimension>0 and an existing docs/ file; every score model has non-empty hf_overrides; no generate model sets a dimension; uv run pytest tests/test_catalog.py passes. File: tests/test_catalog.py.

### t3 — Fleet compose: add vllm-embed (--task embed + MRL hf-overrides) and vllm-rerank (--task score + Qwen3ForSequenceClassification hf-overrides) services to the fleet template, and add their gateway env wiring; set GPU mem utils to sum < 1.0 alongside the 27B primary.

- covers: c1, c3
- acceptance:
  - templates/fleet/docker-compose.yml defines vllm-embed + vllm-rerank services (correct model, --task, --hf-overrides, expose:8000, healthcheck) and adds EMBED_URL/EMBED_SERVED_NAME + RERANK_URL/RERANK_SERVED_NAME to the gateway environment; env.example documents the new vars; the three GPU utils sum < 1.0; docker compose config parses. Files: templates/fleet/docker-compose.yml, templates/fleet/env.example.
  - embed + rerank services set a small --max-model-len (e.g. 8192) and small --gpu-memory-utilization so KV cache stays tiny and all three backends co-fit the GB10.

### t4 — Gateway config: build_config in gateway/_config.py learns an embed backend (EMBED_URL/EMBED_SERVED_NAME) and a rerank backend (RERANK_URL/RERANK_SERVED_NAME), added to the RoutingTable only when their env is present, so resolve_model routes those served names to their backends.

- covers: c11
- acceptance:
  - build_config appends an embed and/or rerank Backend when the env vars are set; RoutingTable.backends includes them; existing primary/fallback behavior is byte-for-byte unchanged when the new vars are absent; unit test passes with a dict env. File: model_gear/gateway/_config.py.

### t5 — Gateway routing verification + minimal fix: prove handle_post forwards POST /v1/embeddings, /v1/score, /v1/rerank to the backend named by the body's model field; document the before-state (no warm embed backend); add a server.py fix ONLY if a failing test proves one is needed; chat + audio routing unchanged.

- depends on: t4
- covers: c4, h1, h4, h9
- acceptance:
  - tests/ post embed/score/rerank bodies (each carrying model) through handle_post and assert they route to the embed/rerank backend and rewrite model to the served name; a test documents that without an embed backend the request 502s/own-errors (before-state); /v1/models lists embed+score served names; chat & /v1/audio tests still pass; any server.py change has a test that fails without it. Files: tests/test_gateway_routing.py (+ gateway/server.py iff justified).
  - an embed/score request is NOT failed over to a chat backend (task-aware / owner-only failover), so a down embed backend gives a clear error rather than a confusing 400 from the 27B; chat failover unchanged.

### t6 — model switch --task: add a --task {generate,embed,score} flag that writes VLLM_TASK and surfaces the per-model hf-overrides as a compose-edit notice (same mechanism as the MoE/MTP serve-extras), so an embed or score gear can be served solo in the single-model deployment for isolated load-testing.

- depends on: t1
- covers: c7
- acceptance:
  - model switch <embed-model> --apply writes VLLM_TASK=embed and prints the hf-overrides notice; the single-model template threads  into the vllm command; switch dry-run shows the plan; existing chat switch output unchanged when task=generate; unit test covers the new flag. Files: model_gear/cli/_commands/switch.py, model_gear/templates/docker-compose.yml.
  - for --task embed|score, switch uses a small max-model-len + gpu-mem-util and skips the tool-calling probe (embed/score have no tool calling).

### t7 — Document the call shapes: add model explain embeddings / rerank / score entries (request+response+dimension+served names) and write per-model docs docs/qwen3-embedding-0.6b.md + docs/qwen3-reranker-0.6b.md (OpenAI-style usage, composition with eidetic, no model-gear SDK).

- depends on: t1
- covers: c2, c5, c12, h2, h5, h10
- acceptance:
  - model explain embeddings, model explain rerank, model explain score each resolve to a body documenting the exact request/response + dimension + served model name; docs/qwen3-embedding-0.6b.md and docs/qwen3-reranker-0.6b.md exist with a placeholder for measured numbers; documented served names equal the catalog ids; markdownlint clean. Files: model_gear/explain/catalog.py, docs/qwen3-embedding-0.6b.md, docs/qwen3-reranker-0.6b.md.
  - the explain _GATEWAY entry drops the 'nothing warm behind /v1/embeddings' line and the _MODELS entry lists the two new per-model docs.

### t8 — Boundary guard: a test asserting model-gear adds no vector store / index / chunker / retrieval — pyproject dependencies are unchanged (still no vector-DB deps) and no new module under model_gear/ implements storage/retrieval.

- covers: c6, h6
- acceptance:
  - tests/ assert the diff adds no vector-DB dependency to pyproject.toml and no module named like store/index/retrieve/chunk under model_gear/; test passes. File: tests/test_boundary_serves_only.py.

### t9 — Load-test both gears WARM on the DGX Spark: bring up the fleet, curl /v1/embeddings (assert 1024-len vector) and /v1/rerank (assert sorted relevance_score) while /v1/chat/completions still answers from the 27B; paste the measured dimension + throughput + co-residency numbers into the per-model docs.

- depends on: t3, t4, t5, t6, t7
- covers: c7, c13, h1, h3, h7, h11
- acceptance:
  - with the fleet up on the GB10: curl /v1/embeddings returns data[0].embedding of length 1024; curl /v1/rerank returns results sorted by relevance_score; /v1/chat/completions returns a 27B completion in the same window; the three containers are simultaneously healthy; docs/qwen3-embedding-0.6b.md + docs/qwen3-reranker-0.6b.md carry the real measured numbers (not placeholders). Files: docs/qwen3-embedding-0.6b.md, docs/qwen3-reranker-0.6b.md (+ optional smoke note).

## Risks

- [unknown_nonblocking] vLLM nv26.04 image (vllm:26.04-py3) is assumed to honor --task embed and --task score; if it does not, the serve flags change — verified empirically at the t9 load-test. (task t9)
