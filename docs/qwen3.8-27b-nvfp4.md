# unsloth/Qwen3.8-27B-NVFP4 — the multimodal cortex, VALIDATED at 1M

The fleet's default primary since 2026-08-19 (plan
`docs/plans/2026-08-19-qwen3-8-cortex-upgrade.md`), replacing
`unsloth/Qwen3.6-27B-NVFP4` (demoted to candidate, kept per cite-don't-delete).
**VALIDATED live on the DGX Spark GB10** — evidence:
`docs/evidence/2026-08-19-accept-qwen38-1m-spark.txt` (boot + gates),
`docs/evidence/2026-08-19-spike-qwen3.8-official-nightly-spark.txt` (engine
spike), `docs/evidence/2026-08-19-baseline-qwen3.6-cortex-spark.txt` (the
incumbent baseline it was measured against).

## Checkpoint facts (config-verified 2026-08-19, snapshot 7d6f8d4d)

- `architectures: ["Qwen3_5ForConditionalGeneration"]`, `model_type: qwen3_5` —
  the same architecture family as the 3.6 it replaced, so the swap was a
  checkpoint change, not a new-arch bring-up.
- 262,144 native context; 64 layers, hybrid linear-attention; **1M
  (1,048,576) served via YaRN** `hf-overrides` (below).
- **Multimodal** — own ViT (image + video intake); no `--language-model-only`.
- **Self-hosted MTP draft**: a separate 849 MB `model_mtp.safetensors` module;
  the generic `{"method": "mtp"}` speculative config applies, no external
  draft repo, no `qwen3_5_mtp` method key needed on vLLM >= 0.26.
- Quantization: compressed-tensors, mixed-precision (fp8 attention + nvfp4
  MLP; ViT unquantized). ~23.4 GB weights (single main shard + MTP module —
  a different layout from the 3.6's five shards).
- Ships its own tokenizer (**re-verify `tokenizer.json` `"truncation": null`
  after every download** — an early revision hardcoded silent 2048-token
  truncation; the current revision is fixed, verified locally).
- `chat_template.jinja` carries `preserve_thinking` — the issue #93 flag
  works unchanged.
- Tool parser: `qwen3_coder` family; the `qwen3_coder_thinking` strict-tools
  plugin (colleague#320) loads and behaves on vLLM 0.26.1 — proven live with a
  `strict: true` + `enable_thinking` call returning a schema-valid tool_call.

## Engine

`vllm/vllm-openai@sha256:8bd082c274fae025b7079498fe1da65182ba1d4c2188c0f5a68c1042c38c3695`
— the **official** Docker Hub nightly (built 2026-08-19), resolved vLLM
`0.26.1rc1.dev942+g5a4c8d992`. The NVIDIA forum thread (380244) claimed stock
`vllm-openai` lacks NVFP4 kernels for sm_121a and required a custom build —
**disproven**: this stock nightly serves the checkpoint on the GB10 with
FlashInfer autotuning under an `121a` arch path. No third-party image was
needed; official-first order was followed (operator decision q3).

## Serving shape (spark-lobe, MEASURED 2026-08-19)

| knob | value | provenance |
|---|---|---|
| `gpu_mem_util` | **0.58** | 0.60 refused twice live (73.01 GiB demanded vs 72.22/65.22 free); 0.58 booted after the embed-deep reclaim |
| `max_model_len` | **1,048,576** | YaRN ×4 over the 262,144 native ceiling |
| `hf-overrides` | `{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144,…}}}` | stock rope fields preserved byte-for-byte; only type/factor/original added |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | `1` | vLLM refuses past the declared ceiling otherwise |
| KV pool | 42.07 GiB = 1,271,476 tokens | **1.21× ceiling at full 1M** — effectively single-request at max depth |
| speculative | `{"method":"mtp","num_speculative_tokens":2}` | 54–61% draft acceptance at n=2 |
| co-residency cost | the opt-in `embed-deep` 4B gear is **stopped** on this box | operator reclaim decision (spec q4) |

Rollback pair (measured 2026-07-31, retained in `spark-lobe.toml`):
`gpu_mem_util=0.44` / `max_model_len=262144`, no YaRN knobs.

## Measured performance (single-stream, via gateway, `usage.completion_tokens`)

| shape | TTFT | decode tok/s | 3.6 baseline (same day) |
|---|---|---|---|
| short | 0.20–0.25 s | 23.6–24.7 | 26.5–31.0 |
| medium | 0.35 s | 22.0–22.4 | 26.1–29.1 |
| long (700 tok) | 0.25 s | 19.6–20.1 | 23.7–24.1 |

The 1M lane runs ~2–5 tok/s under the 3.6 baseline. At native 262,144 (t2
spike, same engine) it measured 20.6–30.4 — the YaRN window costs ~0–1.5 tok/s
of decode.

Long-context, measured: 228,415-token prompt → ~723 tok/s prefill (316 s);
**328,379-token prompt (beyond native) → ~563 tok/s prefill (583 s), exact
needle retrieval**. Prefill runs *minutes* at depth — consumers should
**stream** near-1M requests; a non-streamed request happened to survive 9.7
silent minutes on this stack, but that is measured luck, not a contract.

## YaRN quality cost (measured, not assumed)

Always-on static YaRN was the challenge pass's biggest unargued assumption
(spec c26). Measured on an 8-prompt deterministic QA set, same engine and
flags: **native 7/8 vs 1M-YaRN 7/8 — identical score, identical single
failure** (a prompt whose thinking exhausts the token budget in both configs).
Zero measurable cost on this set; it is a small set — a broader eval remains
open.

## Known limits

- 1.21× KV ceiling at 1M: one full-depth request effectively owns the lane
  for its multi-minute prefill; no per-request KV fairness in v1 (plan risk r3).
- MTP acceptance at 1M depth beyond the probes above is lightly measured
  (plan risk r2); `num_speculative_tokens` tuning (forum suggests 5 at native)
  is open.
- The digest bump reaches hand/worker **templates** too, but those lanes are
  UNVALIDATED on their hosts (Thor sm_110/Orin) until a future boot there
  (plan risk r1; Spark-only rollout boundary c25).

See `docs/qwen38-rollout-notes.md` for the consumer repoint checklist and
`docs/model-switch-playbook.md` for the swap/rollback procedure.
