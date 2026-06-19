# Qwen3-Embedding-0.6B — embedding gear (1024-dim)

> One entry in model-gear's **supported catalog** (`model overview --list`). For
> the catalog-vs-warm distinction — what you *can* load vs. what's loaded *now* —
> see [`gateway-fleet.md`](gateway-fleet.md#supported-catalog-vs-warm-backends).

## What it is

- 0.6B **dense** text embedding model from the Qwen3 family.
- **1024-dim** native output with **Matryoshka Representation Learning (MRL)**
  nesting at 32 / 64 / 128 / 256 / 512 / 768 / 1024 dimensions — consumers can
  request a smaller embedding with the `"dimensions"` request parameter, without
  re-serving the model.
- **32K native** context (`--max-model-len 32768`).
- Served via vLLM's `/v1/embeddings` endpoint (`--task embed` pooling mode).
- No tool parser, no quantization flag — this is a pooling model, not a chat model.
- **Served name == catalog id:** `Qwen/Qwen3-Embedding-0.6B`.

## Serving

Served as a **warm fleet backend** alongside the 27B primary and the reranker on
the DGX Spark GB10 (128 GB unified memory). Its small footprint (0.6B weights,
32K KV window) keeps the KV cache tiny so all three backends co-fit without
memory pressure.

```bash
model switch --model Qwen/Qwen3-Embedding-0.6B --apply
model serve --apply
model status
```

Key compose flags:

- `--task embed` — vLLM embedding (pooling) serving mode
- `--hf-overrides '{"is_matryoshka": true, "matryoshka_dimensions": [32, 64, 128, 256, 512, 768, 1024]}'`
- `--max-model-len 32768`

## API call shapes

The gateway routes `/v1/embeddings` to this backend by matching
`"model": "Qwen/Qwen3-Embedding-0.6B"` — the same gateway port as chat.

> **The `model` field is required.** Routing is by model name, so a request
> without `model` falls through to the gateway's default (the chat primary),
> which can't serve embeddings (returns a 400). The OpenAI embeddings API marks
> `model` required, so a spec-compliant client always sends it.

### Basic embedding (native 1024-dim)

```bash
curl -s http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "input": ["text a", "text b"]
  }'
```

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [/* 1024 floats */]},
    {"object": "embedding", "index": 1, "embedding": [/* 1024 floats */]}
  ],
  "model": "Qwen/Qwen3-Embedding-0.6B",
  "usage": {"prompt_tokens": 4, "total_tokens": 4}
}
```

### MRL-truncated embedding (e.g. 256-dim)

Pass the optional `"dimensions"` parameter to truncate to any supported
Matryoshka sub-dimension:

```bash
curl -s http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "input": "search query here",
    "dimensions": 256
  }'
```

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [/* 256 floats */]}
  ],
  "model": "Qwen/Qwen3-Embedding-0.6B",
  "usage": {"prompt_tokens": 3, "total_tokens": 3}
}
```

Supported `dimensions` values: 32, 64, 128, 256, 512, 768, 1024. Values outside
this set are rejected by the HF override at serve time.

## Health check

```bash
curl -fsS http://localhost:8000/health
curl -s http://localhost:8000/v1/models
```

## Co-residency note

This backend runs **warm alongside the 27B primary** (`Qwen3.6-27B-Text-NVFP4-MTP`)
and the reranker (`Qwen3-Reranker-0.6B`) on the single GB10. Because it uses a
pooling task (no autoregressive decode), its KV cache footprint is negligible —
it does not compete with the primary for KV memory even under concurrent
embedding workloads.

The gateway routes requests by `model` field at the shared port, so embedding
and chat calls share one endpoint with zero client-side configuration.

## Composition with eidetic-cli

model-gear vectorizes text via this backend; eidetic-cli stores and retrieves
the resulting vectors. Typical pipeline:

1. `POST /v1/embeddings` (model-gear) → 1024-dim vector(s)
2. eidetic ingest — stores vectors + metadata in its index
3. eidetic retrieve — nearest-neighbour search returns candidate chunks
4. optionally re-rank candidates with `Qwen/Qwen3-Reranker-0.6B` via `/v1/rerank`

## Assessment / Benchmark

<!-- measured numbers pasted in by the load-test (#44) -->

Run `model assess` after serving to measure correctness. Run `model benchmark`
for throughput. The assessment suite probes `/v1/embeddings` and reports
tokens/s, embedding dimension, and (if `--tools` is passed) tool-call behaviour
(not applicable to this pooling model — `--tools` is silently skipped).
