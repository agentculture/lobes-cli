# unsloth/Qwen3.8-27B-NVFP4 — the multimodal cortex candidate (promotion in progress)

> **STATUS: PENDING VALIDATION.** This doc is a stub created when the catalog
> entry landed (plan `qwen3-8-cortex-upgrade`, task t3 merge). Task t9 replaces
> this with the full per-model doc once the live GB10 boot (t7) and acceptance
> gates (t8) land their evidence under `docs/evidence/`. Per the #108 rule,
> nothing here may be read as VALIDATED until that transcript exists.

## Config-verified facts (fetched 2026-08-19)

- `architectures: ["Qwen3_5ForConditionalGeneration"]`, `model_type: qwen3_5` —
  the same architecture id as the outgoing `unsloth/Qwen3.6-27B-NVFP4` primary.
- `text_config.max_position_embeddings = 262144` (native 256K), 64 layers,
  hybrid linear-attention; 1M (1048576) reachable via YaRN `hf-overrides` on
  `text_config.rope_parameters` (preserve every mrope field).
- `mtp_num_hidden_layers = 1` — self-hosted MTP draft module, no external repo.
- `vision_config` present (multimodal — do NOT pass `--language-model-only`);
  no audio config.
- `quantization_config`: `compressed-tensors`, format `mixed-precision`
  (fp8 attention + nvfp4 MLP; ViT unquantized).
- `tokenizer.json` `"truncation": null` in the current revision — an early
  revision hardcoded silent 2048-token truncation; RE-VERIFY after download.
- `chat_template.jinja` carries `preserve_thinking` (issue #93 flag survives).
- License apache-2.0; ~23.4 GB across 5 shards.

## Baseline to beat

The incumbent 3.6 was benchmarked live on 2026-08-19 before this swap:
`docs/evidence/2026-08-19-baseline-qwen3.6-cortex-spark.txt`.

## Serving knobs

Measured values land here after the t7 boot. Do not copy declared numbers as
measured (`docs/machine-profiles.md` rule).
