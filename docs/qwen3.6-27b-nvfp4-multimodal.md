# Qwen3.6-27B-NVFP4 (unsloth) — the multimodal `cortex` candidate

**Status: `configured` — DECLARED, NEVER BOOTED.**

Everything on this page is read off the checkpoint's own published config files
(fetched 2026-07-31). Nothing here comes from a live run. No `gpu_mem_util` and
no `max_model_len` is declared anywhere for this model, deliberately — on a
unified-memory card those are *measured* truths, not arithmetic. That rule was
established the hard way twice: `thor-muse`'s hypothesised `0.40` was refused by
vLLM on the physical box (`0.55` measured), and `thor-worker`'s `0.45` booted
first try only because the MoE's active-parameter footprint is small. See
[`docs/machine-profiles.md`](machine-profiles.md).

Promotion to `role_hint="primary"` is gated on a live GB10 boot plus an
acceptance transcript under `docs/evidence/`.

## Why this checkpoint

It would give the fleet's reasoning / deciding / final-authority lobe **image and
video intake for the first time**. Today `cortex` serves
`sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`, whose export physically removed the
ViT tower — so every visual question must be delegated to `senses` (perception
only, forbidden from `final_decision` / `repo_action` / `security_decision`) or
to `worker` (a doer, forbidden from `final_decision` / `security_decision`).
Neither may make the call. The contract as written has **no role that both sees
an image and decides**.

The strongest argument for this particular export is that it is a **structural
twin of the already-validated `worker` gear**,
[`unsloth/Qwen3.6-35B-A3B-NVFP4`](qwen3.6-35b-a3b-nvfp4.md) — same publisher,
same mixed-precision compressed-tensors recipe, same self-hosted MTP draft, same
ViT-kept multimodal shape. That gear booted healthy on the physical Thor at the
first attempt. This is the best prior available short of a boot; it is **not**
evidence.

## Verified from the published config

| | value | source |
|---|---|---|
| architecture | `Qwen3_5ForConditionalGeneration`, `model_type: qwen3_5` | `config.json` |
| context | `text_config.max_position_embeddings = 262144` (256K native) | `config.json` |
| layers | 64, hybrid linear-attention | `config.json` |
| **vision** | `language_model_only: false`; `vision_config` 27-layer ViT (hidden 1152) | `config.json` |
| **image + video** | `image_token_id=248056`, `video_token_id=248057`, `vision_start/end=248053/248054` | `config.json` |
| audio | **none** — no `audio_config` | `config.json` |
| **MTP** | `mtp_num_hidden_layers=1`, top-level `unsloth_fixed_mtp` flag, **15 real `mtp.*` tensors** | `config.json`, `model.safetensors.index.json` |
| quantization | `compressed-tensors`, `format: mixed-precision` — fp8 attn/`lm_head`/upper-8 MLP, nvfp4 MLP gate/up/down; ViT and every `linear_attn` left **unquantized** (303 ignore patterns) | `config.json` |
| size | 1968 tensors, **23.42 GB**, 5 shards | `model.safetensors.index.json` |
| `preserve_thinking` | **present** in the template | `chat_template.jinja` |
| tokenizer | ships its own | repo listing |
| license | Apache-2.0 | model card |

## What a promoted lane would have to change

The primary compose lane converges onto the **already-validated `worker` lane's
shape** — same publisher, same recipe:

| flag | today (text-only primary) | promoted |
|---|---|---|
| `--quantization` | `modelopt` (→ `modelopt_fp4`) | **`compressed-tensors`** |
| `--speculative-config` | `{"method": "qwen3_5_mtp", "num_speculative_tokens": 3}` (grafted draft) | **`{"method": "mtp", "num_speculative_tokens": 2}`** (self-hosted) |
| `--language-model-only` | present (ViT removed) | **dropped** |
| `--tokenizer=mmangkad/…` | present (TokenizersBackend workaround) | **dropped** — ships its own |

**Kept unchanged**, and this is load-bearing:
`--trust-remote-code`, `--enable-prefix-caching`, `--tool-call-parser=qwen3_coder`,
`--reasoning-parser=qwen3`, `--default-chat-template-kwargs '{"preserve_thinking": true}'`
(issue #93 — the variable exists in this checkpoint's template, verified above),
and `--tool-parser-plugin=/opt/lobes/qwen3_thinking_tool_parser.py` with
`GATEWAY_FORCE_STRICT_TOOLS` armed (colleague#320). A promotion that silently
drops either is a regression, and the acceptance run must re-prove both.

## Knock-on effect if promoted

`mmangkad/Qwen3.6-27B-NVFP4` is currently justified in `CLAUDE.md` as (a) the
tokenizer source the MTP primary serves with and (b) the only vision-capable
27B. Promotion makes **both false**. The entry stays (cite-don't-delete), but its
recorded rationale must be rewritten rather than left asserting something untrue.

## Open, unmeasured

- **Budget.** The incumbent runs `PRIMARY_GPU_MEM_UTIL=0.44` at the full 262144
  window — measured for a *text-only* checkpoint. This export adds an
  unquantized bf16 ViT (333 visual tensors) plus video preprocessing. Whether
  the full 256K window survives, whether `embedder`/`reranker`/`embed-deep`
  still co-reside, and whether `PRIMARY_MAX_NUM_SEQS` needs changing are all
  boot-time findings.
- **MTP acceptance.** The 35B twin hit 89.1% at 2 tokens. That is the sibling's
  number. This checkpoint's is unknown, as is whether 2 or 3 tokens is right.
- **Vision quality at 27B.** Nothing is known about how this export's ViT
  performs relative to the 12B `senses` gear it would sit alongside.

## See also

- [`qwen3.6-35b-a3b-nvfp4.md`](qwen3.6-35b-a3b-nvfp4.md) — the validated
  structural twin (`worker`)
- [`qwen3.6-27b-text-nvfp4-mtp.md`](qwen3.6-27b-text-nvfp4-mtp.md) — the
  incumbent text-only primary
- [`machine-profiles.md`](machine-profiles.md) — why budgets are measured
