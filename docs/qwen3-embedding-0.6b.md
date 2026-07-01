# Qwen3-Embedding-0.6B — embedding gear (1024-dim)

> One entry in lobes's **supported catalog** (`lobes overview --list`). For
> the catalog-vs-warm distinction — what you *can* load vs. what's loaded *now* —
> see [`gateway-fleet.md`](gateway-fleet.md#supported-catalog-vs-warm-backends).
>
> **On vLLM nightly since the fleet-wide nightly-unification migration**
> (`docs/vllm-nightly-migration.md` §3/§5, t3/t4) — this gear now runs the same
> pinned `vllm/vllm-openai@sha256:7c5a10e9...` digest (vLLM `0.23.1rc1.dev672`)
> as the primary and multimodal gears, not the `nv26.04-py3` / `0.19.0` build
> the 2026-06-19 benchmark below was measured on. The nightly t3 spike (§5)
> re-confirmed `/v1/embeddings` → 1024-dim, matryoshka `--hf-overrides`
> accepted, `--runner pooling --convert embed` unchanged — no serving-flag
> drift. The benchmark numbers below remain the historical 0.19.0 record.

## What it is

- 0.6B **dense** text embedding model from the Qwen3 family.
- **1024-dim** native output with **Matryoshka Representation Learning (MRL)**
  nesting at 32 / 64 / 128 / 256 / 512 / 768 / 1024 dimensions — consumers can
  request a smaller embedding with the `"dimensions"` request parameter, without
  re-serving the model.
- **32K native** context, served at `--max-model-len 8192` (tiny KV footprint).
- Served via vLLM's `/v1/embeddings` endpoint in pooling mode
  (`--runner pooling --convert embed`; unchanged since the fleet's move to
  vLLM nightly — see the note above).
- No tool parser, no quantization flag — this is a pooling model, not a chat model.
- **Served name == catalog id:** `Qwen/Qwen3-Embedding-0.6B`.

## Serving

Served as a **warm fleet backend** alongside the 27B primary and the reranker on
the DGX Spark GB10 (128 GB unified memory). Its small footprint (0.6B weights,
32K KV window) keeps the KV cache tiny so all three backends co-fit without
memory pressure.

The warm path is the **fleet** (`lobes init --fleet` then `lobes fleet up --apply`),
which brings up `vllm-embed` + `vllm-rerank` + the primary behind one gateway. To
serve it *solo* for isolated testing, `lobes switch Qwen/Qwen3-Embedding-0.6B` (the
task is auto-detected from the catalog) prints the exact compose edits to apply.

Key compose flags:

- `--runner pooling --convert embed` — vLLM pooling/embedding mode on this build
  (replaces the old `--task embed`, which is rejected as an unknown argument)
- `--hf-overrides '{"is_matryoshka": true, "matryoshka_dimensions": [32, 64, 128, 256, 512, 768, 1024]}'`
- `--max-model-len 8192`
- `--gpu-memory-utilization 0.06` — 0.025 fails with "No available memory for the
  cache blocks" (the pooling runner still reserves a cache-block budget)

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

lobes vectorizes text via this backend; eidetic-cli stores and retrieves
the resulting vectors. Typical pipeline:

1. `POST /v1/embeddings` (lobes) → 1024-dim vector(s)
2. eidetic ingest — stores vectors + metadata in its index
3. eidetic retrieve — nearest-neighbour search returns candidate chunks
4. optionally re-rank candidates with `Qwen/Qwen3-Reranker-0.6B` via `/v1/rerank`

## Assessment / Benchmark

**Load-tested 2026-06-19 on the DGX Spark (GB10, 128 GB unified)** — served warm
under `--runner pooling --convert embed`, `--gpu-memory-utilization 0.06`,
`--max-model-len 8192`, **co-resident with the 27B primary and the reranker** (all
three backends simultaneously healthy on the one GB10):

| Metric | Result |
|---|---|
| embedding dimension | **1024** (native) |
| MRL truncation | `"dimensions": 256` → 256-dim vector ✓ (32/64/128/256/512/768/1024) |
| single-text latency (warm) | ~28 ms |
| batch throughput | 16 texts ≈ 107 ms (~150 texts/s) |
| co-residency | 27B chat still answered normally while the embedder served |

Served on this vLLM build (`0.19.0+nv26.04`) with `--runner pooling --convert embed`
— the older `--task embed` is rejected (`unrecognized arguments`). The probes use
plain `curl` against `/v1/embeddings` (the `lobes assess` arithmetic/tool probes are
chat-oriented and not applicable to a pooling model).

This flag set is unchanged on the fleet's current vLLM nightly build
(`docs/vllm-nightly-migration.md` §5, t3 spike, live 2026-07-01) —
`--runner pooling --convert embed`, the matryoshka `--hf-overrides`, and the
1024-dim output all still work identically; the numbers above were not
re-measured on nightly but the serving contract did not change.
