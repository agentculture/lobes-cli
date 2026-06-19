# Qwen3-Reranker-0.6B — reranker gear (score / rerank)

> One entry in model-gear's **supported catalog** (`model overview --list`). For
> the catalog-vs-warm distinction — what you *can* load vs. what's loaded *now* —
> see [`gateway-fleet.md`](gateway-fleet.md#supported-catalog-vs-warm-backends).

## Overview

- 0.6B **dense cross-encoder** from the Qwen3 family — a `Qwen3ForSequenceClassification`
  model with a binary **yes / no** logit head.
- Scores (query, passage) pairs for retrieval re-ranking via vLLM's `/v1/score` endpoint
  (`task="score"` in the catalog).
- **32K native** context (`max_model_len 32768`).
- No tool parser and no quantization flag — this is a scoring model, not a chat model.
- The `hf_overrides` declare the non-standard architecture class and classifier tokens
  so vLLM can load the classification head correctly.

## Serving

```bash
model switch --model Qwen/Qwen3-Reranker-0.6B --apply
model serve --apply
model status
```

Key compose flags:

- `--task score` — vLLM cross-encoder scoring mode
- `--hf-overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}'`
- `--max-model-len 32768`

Verify:

```bash
curl -fsS http://localhost:8000/health
curl -s http://localhost:8000/v1/models
curl -s http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-Reranker-0.6B", "text_1": "What is AI?", "text_2": "Artificial intelligence is..."}'
```

## Assessment

<!-- measured numbers pasted in by the load-test (#44) -->

Run `model assess` after serving to measure throughput and correctness.
The assessment suite probes the `/v1/score` endpoint and reports
tokens/s and score distribution over a reference (query, passage) pair set.
