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

## MEASURED — live GB10 boot, 2026-07-31

The candidate booted on the DGX Spark and was probed live. These are measured
numbers, not hypotheses.

**Budget — booted first try at the incumbent's own knobs**, no retune needed
(unlike `thor-muse`, whose 0.40 was refused):

| | incumbent (text-only) | candidate (multimodal) |
|---|---|---|
| `gpu_mem_util` / `max_model_len` | 0.44 / 262144 | **0.44 / 262144** |
| KV cache | — | **26.39 GiB** |
| KV pool | 888,946 tokens | **756,642 tokens** |
| concurrency @ full context | 3.39× | **≈2.89×** |

The unquantized bf16 ViT costs ≈**132,300 tokens of KV pool (~15%)**. The full
256K window survives, and `embedder`, `reranker` and `embed-deep` all stayed
co-resident and healthy throughout.

**Throughput** (single-stream, `enable_thinking=false`, measured against
`usage.completion_tokens`):

| case | TTFT | decode tok/s |
|---|---|---|
| short prompt / 21 tok gen | 0.262 s | 14.9 |
| medium prompt / 191 tok gen | 0.275 s | 16.4 |
| long gen / 512 tok | 0.267 s | **19.0** |

**MTP self-hosted draft engages** — mean acceptance length 2.24–2.35,
per-position rate ~0.79/0.52, avg draft acceptance **62–67%**. That resolves the
previously-unmeasured question: the self-hosted head works at
`num_speculative_tokens: 2`. It is lower than both the 35B twin's 89.1% and the
incumbent's recorded 72–78.6%.

**Behavioural gates — all pass:**

- **Vision**: red/blue squares with an opposite-colour negative control.
- **Video**: a white square crossing a black field, asked left-to-right vs
  right-to-left, with the *reversed* clip as the control — both correct. A
  single-frame read cannot pass this, so temporal processing is real.
- **Thinking**: 4,195 chars of trace on a deliberately misleading puzzle.
  *Note the field is `reasoning`, NOT `reasoning_content`* — see the playbook.
- **`preserve_thinking` (#93)**: two-turn `prompt_tokens` delta of **+800**,
  i.e. historical `<think>` is retained across turns.
- **Strict tool calling with thinking on (colleague#320)**: clean
  `read_file {"path": "calc.py"}`, `finish_reason: tool_calls` — no 500 grammar
  rejection, no mangled name.

### The comparison is NOT controlled

The incumbent's 18.7–19.1 tok/s and 72–78.6% acceptance were recorded on a
**different vLLM build** (0.19.0+nv26.04) at a different util. Sustained decode
being "level with the incumbent" is therefore a weaker claim than it reads. A
fair head-to-head means re-benchmarking the incumbent on today's engine before
promoting. See [`model-switch-playbook.md`](model-switch-playbook.md) §1.

## Still open

- **Quality.** Nothing above measures whether this model *reasons better* — only
  that it is alive, fast and structurally intact. A swap that degraded reasoning
  quality would pass every gate here.
- **Concurrency.** All numbers are single-stream; ~2.89× is implied by the KV
  pool, not measured.
- **Long context.** The lane serves 262144 tokens; the longest probe used ~94.
- **Vision quality at 27B** relative to the 12B `senses` gear it sits alongside.
- **Breaking change.** Promotion changes the served id, and **no consumer in the
  mesh addresses by role name** — every one sends the raw id and would 404. See
  the playbook §2.

## See also

- [`qwen3.6-35b-a3b-nvfp4.md`](qwen3.6-35b-a3b-nvfp4.md) — the validated
  structural twin (`worker`)
- [`qwen3.6-27b-text-nvfp4-mtp.md`](qwen3.6-27b-text-nvfp4-mtp.md) — the
  incumbent text-only primary
- [`machine-profiles.md`](machine-profiles.md) — why budgets are measured
