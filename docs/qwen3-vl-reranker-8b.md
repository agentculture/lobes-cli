# Qwen3-VL-Reranker-8B — the `qwen3vl-rerank` EXCLUDED lane

> One entry in lobes's supported catalog. **Status: `configured` — it never
> served on the Jetson AGX Orin and is EXCLUDED by decision c18.** Transcript:
> [vLLM spike](evidence/2026-10-07-spike-orin-embed-vllm-8b.txt).

## Catalog entry

`Qwen/Qwen3-VL-Reranker-8B`, lane name `qwen3vl-rerank` (task `score`):
multimodal reranker (text, image, video), 32768 context, no embedding
dimension. `role_hint="candidate"` (never `reranker`; the 0.6B reranker role is
untouched).

## License

Apache-2.0, per the Hugging Face card read 2026-10-07 (PUBLISHED-ELSEWHERE).

## What was measured on the Orin (2026-10-07)

It did **not** serve at util 0.35 (KV -6.17 GiB) with the card's recipe
(`hf_overrides` `Qwen3VLForSequenceClassification`, `classifier_from_token`
`no,yes`, `is_original_qwen3_reranker`, the chat template from the image). A
retry at a 44g cap failed with "No available memory for the cache blocks". It has
the same encoder-profile cost as the embedder (about 0.5 util) and was not
retried higher.

## Why it is excluded

Decision c18 admitted it only if the measured Orin budget fits it beside the
embedders. About 0.5 util for the reranker plus at least 0.35 for any 8B
embedder, EG2 (about 4 GiB RSS) and the 0.6B pair (0.12) exceeds the board. The
8B reranker is **excluded on measurement**; the 0.6B reranker stays.

## Status in the repo

The catalog entry, the lane registry row and a compose service remain in-tree
(cite-don't-delete) with DECLARED, unmeasured defaults, so they must not be read
as a working recipe. The compose service's own comment notes the card's vLLM
recipe flags were not applied. Nothing was quality-tested: this model never
produced a score.
