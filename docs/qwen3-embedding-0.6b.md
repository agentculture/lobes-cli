# Qwen3-Embedding-0.6B — embedding gear (1024-dim)

> One entry in model-gear's **supported catalog** (`model overview --list`). For
> the catalog-vs-warm distinction — what you *can* load vs. what's loaded *now* —
> see [`gateway-fleet.md`](gateway-fleet.md#supported-catalog-vs-warm-backends).

## Overview

- 0.6B **dense** text embedding model from the Qwen3 family.
- **1024-dim** output with **Matryoshka** nesting at 32 / 64 / 128 / 256 / 512 / 768 / 1024 dimensions —
  consumers can request a sub-1024 dimension without re-serving the model.
- **32K native** context (`max_model_len 32768`).
- Served via vLLM's `/v1/embeddings` endpoint (`task="embed"` in the catalog).
- No tool parser and no quantization flag — this is a pooling model, not a chat model.

## Serving

```bash
model switch --model Qwen/Qwen3-Embedding-0.6B --apply
model serve --apply
model status
```

Key compose flags:

- `--task embed` — vLLM embedding serving mode
- `--hf-overrides '{"is_matryoshka": true, "matryoshka_dimensions": [32, 64, 128, 256, 512, 768, 1024]}'`
- `--max-model-len 32768`

Verify:

```bash
curl -fsS http://localhost:8000/health
curl -s http://localhost:8000/v1/models
curl -s http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-Embedding-0.6B", "input": "hello world"}'
```

## Assessment

<!-- measured numbers pasted in by the load-test (#44) -->

Run `model assess` after serving to measure throughput and correctness.
The assessment suite probes the `/v1/embeddings` endpoint and reports
tokens/s and embedding dimension.
