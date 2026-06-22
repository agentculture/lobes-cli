# Qwen3-Reranker-0.6B — reranker gear (score / rerank)

> One entry in lobes's **supported catalog** (`lobes overview --list`). For
> the catalog-vs-warm distinction — what you *can* load vs. what's loaded *now* —
> see [`gateway-fleet.md`](gateway-fleet.md#supported-catalog-vs-warm-backends).

## What it is

- 0.6B **dense cross-encoder** from the Qwen3 family — a
  `Qwen3ForSequenceClassification` model with a binary **yes / no** logit head.
- Scores (query, passage) pairs for retrieval re-ranking.
- **32K native** context, served at `--max-model-len 8192` (tiny KV footprint).
- Served via vLLM's pooling/scoring mode (`--runner pooling --convert classify` on
  the nv26.04 build) — one backend handles both `/v1/rerank` (Jina/Cohere shape,
  sorted best-first) and `/v1/score` (raw pairwise scores, input order).
- No tool parser, no quantization flag — this is a scoring model, not a chat model.
- **Served name == catalog id:** `Qwen/Qwen3-Reranker-0.6B`.

## Serving

Served as a **warm fleet backend** alongside the 27B primary and the embedder on
the DGX Spark GB10 (128 GB unified memory). Its small footprint (0.6B weights,
32K KV window) keeps the KV cache tiny so all three backends co-fit.

The warm path is the **fleet** (`lobes init --fleet` then `lobes fleet up --apply`).
To serve it *solo* for testing, `lobes switch Qwen/Qwen3-Reranker-0.6B` (the task is
auto-detected from the catalog) prints the exact compose edits to apply.

Key compose flags:

- `--runner pooling --convert classify` — vLLM scoring mode on this build (replaces
  the old `--task score`; vLLM auto-resolves `--convert auto` to `classify` for a
  `*ForSequenceClassification` arch, but pass it explicitly to silence the notice)
- `--hf-overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}'`
- `--max-model-len 8192`
- `--gpu-memory-utilization 0.06`

## API call shapes

The gateway routes `/v1/rerank` and `/v1/score` to this backend by matching
`"model": "Qwen/Qwen3-Reranker-0.6B"` — the same gateway port as chat and
embeddings.

> **The `model` field is required.** Routing is by model name, so a request
> without `model` falls through to the gateway's default (the chat primary),
> which can't score (returns a 400). Always send `model` in the request body.

### Rerank (Jina / Cohere shape — sorted best-first)

Use `/v1/rerank` when you want results ranked from most to least relevant. The
`index` in each result refers to the position in the original `documents` list.

```bash
curl -s http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Reranker-0.6B",
    "query": "What is the capital of France?",
    "documents": [
      "Paris is the capital of France.",
      "Berlin is the capital of Germany.",
      "Rome is the capital of Italy."
    ]
  }'
```

```json
{
  "results": [
    {"index": 0, "relevance_score": 0.91},
    {"index": 2, "relevance_score": 0.18},
    {"index": 1, "relevance_score": 0.07}
  ]
}
```

Results are sorted **best-first** (highest `relevance_score` first).

### Score (vLLM pairwise shape — input order)

Use `/v1/score` when you need raw scores in the original input order (e.g. to
join scores back to your document list by index without re-sorting).

```bash
curl -s http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Reranker-0.6B",
    "text_1": "What is the capital of France?",
    "text_2": [
      "Paris is the capital of France.",
      "Berlin is the capital of Germany."
    ]
  }'
```

```json
{
  "object": "list",
  "data": [
    {"index": 0, "score": 0.91},
    {"index": 1, "score": 0.07}
  ]
}
```

Results are returned in **input order** (no sorting). Use `/v1/rerank` for
sorted output with the Jina/Cohere interface.

## Health check

```bash
curl -fsS http://localhost:8000/health
curl -s http://localhost:8000/v1/models
```

## Co-residency note

This backend runs **warm alongside the 27B primary** (`Qwen3.6-27B-Text-NVFP4-MTP`)
and the embedder (`Qwen3-Embedding-0.6B`) on the single GB10. Because it uses a
classification task (no autoregressive decode), its KV cache footprint is
negligible — it does not compete with the primary for KV memory even under
concurrent reranking workloads.

The gateway routes requests by `model` field at the shared port, so reranking,
scoring, embedding, and chat calls all share one endpoint with zero client-side
port configuration.

## Composition with eidetic-cli

Typical RAG pipeline using both embed + rerank gears:

1. `POST /v1/embeddings` (Qwen3-Embedding-0.6B) → 1024-dim vectors
2. eidetic ingest — stores vectors + metadata
3. eidetic retrieve — nearest-neighbour search returns top-K candidates
4. `POST /v1/rerank` (Qwen3-Reranker-0.6B) — cross-encoder reranks top-K,
   returning the globally best passages before generation

The embedder handles recall; the reranker handles precision. Both run warm on
the same GB10 gateway port, so neither adds a new service or port to the client.

## Assessment / Benchmark

**Load-tested 2026-06-19 on the DGX Spark (GB10, 128 GB unified)** — served warm
under `--runner pooling --convert classify` + the `Qwen3ForSequenceClassification`
hf-override, `--gpu-memory-utilization 0.06`, `--max-model-len 8192`, **co-resident
with the 27B primary and the embedder** (all three simultaneously healthy):

| Metric | Result |
|---|---|
| endpoints | `/v1/rerank` (sorted) + `/v1/score` (input order) — one backend |
| rerank latency (warm, 1 query × 5 docs) | ~25 ms |
| ranking quality | relevant docs ranked first (e.g. France-capital query: Paris doc top at 0.98) |
| score endpoint | `/v1/score` returns per-pair scores ✓ |
| co-residency | 27B chat unaffected while the reranker served |

Served on this vLLM build (`0.19.0+nv26.04`) with `--runner pooling --convert
classify` — the older `--task score` is rejected (`unrecognized arguments`). The
probes use plain `curl` against `/v1/rerank` and `/v1/score`.
