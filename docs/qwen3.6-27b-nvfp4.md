# Archived former primary: `mmangkad/Qwen3.6-27B-NVFP4`

The **fleet's default primary from 0.10.0 until 2026-05-31**, when it was
superseded by the MTP build
[`sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`](qwen3.6-27b-text-nvfp4-mtp.md)
(~2.4× single-stream decode via speculative decoding). It is **retained as a
candidate**, not removed, for two reasons: (1) it is the **tokenizer source** the
MTP primary serves with (`--tokenizer=mmangkad/Qwen3.6-27B-NVFP4` — the MTP
checkpoint's `tokenizer_config` declares a class absent from the nv26.04 image),
and (2) it is the **only vision-capable 27B** in the catalog (the MTP primary is
text-only), so it is the fallback when an image path is needed.

**Load-tested live on DGX Spark (GB10),** first 2026-05-27 and re-confirmed
2026-05-30 — it loads and serves cleanly under the vLLM image lobes already
runs. Tracked by [issue #6](https://github.com/agentculture/lobes-cli/issues/6).

**Warm-up (solo, 2026-05-30): ~423 s (~7 min)** from container start to `/health`
— weight load 160 s (28.25 GiB), profiling/warmup 55 s, then CUDA-graph capture +
KV allocation. Plan for a multi-minute cold start. Decode re-confirmed at
**8.0 tok/s** (batch=1, 512 tokens); prefill 2,015 tokens in 3.29 s.

Source: <https://huggingface.co/mmangkad/Qwen3.6-27B-NVFP4> — public, Apache-2.0.

> One entry in lobes's **supported catalog** (`lobes overview --list`). For
> the catalog-vs-warm distinction — what you *can* load vs. what's loaded *now* —
> see [`gateway-fleet.md`](gateway-fleet.md#supported-catalog-vs-warm-backends).

## What it is

- NVFP4 (NVIDIA ModelOpt) quantization of **`Qwen/Qwen3.6-27B`**.
- `config.json`: `architectures: ["Qwen3_5ForConditionalGeneration"]`,
  `model_type: qwen3_5`, 64 layers, `hidden_size 5120`,
  `max_position_embeddings 262144` (**256K** context), multimodal RoPE
  (`mrope_interleaved`, `mrope_section`).
- **Hybrid attention:** vLLM loads it with linear-attention / Gated-DeltaNet
  (`gdn_linear_attn`) Mamba layers plus periodic full attention — not a plain
  dense transformer like the 32B. It also carries a **ViT multimodal encoder**
  (it is a vision-language model), though text-only chat serves without an image
  path.
- ~20B effective params; **~29 GB on disk** (BF16 / F8_E4M3 / U8 tensors).
  ModelOpt producer `0.42.0rc1.dev107` (a dev/rc build).

## Is it supported here? — Yes, load-tested and serving

The pre-flight check (query the running engine's registry):

```text
$ docker exec model-gear-vllm python3 -c \
  "from vllm.model_executor.models.registry import ModelRegistry; \
   print('Qwen3_5ForConditionalGeneration' in ModelRegistry.get_supported_archs())"
True
```

The `nvcr.io/nvidia/vllm:26.04-py3` image (engine `0.19.0+...nv26.04`) registers
`Qwen3_5ForConditionalGeneration` (plus the MoE and MTP variants) — the exact
architecture this checkpoint declares. The live load (below) confirms it
instantiates, loads weights, and serves with the same compose flags as the 32B
(`--quantization=modelopt_fp4`, `--reasoning-parser=qwen3`).

## How to run (same compose, model override)

```bash
lobes switch mmangkad/Qwen3.6-27B-NVFP4 --port 8001 --max-model-len 32768 --apply
# (switch is dry-run without --apply; it rewrites VLLM_MODEL / VLLM_SERVED_NAME /
#  VLLM_PORT in .env, auto-selects VLLM_TOOL_CALL_PARSER=qwen3_coder for this
#  model (override with --tool-call-parser), recreates the container, waits for
#  /health, then probes tool_choice:auto to confirm tool calling. qwen3_coder is
#  required for tool calling on this model; see caveat 1.)
```

`VLLM_SERVED_NAME` must match the part after `vllm-local/` in `culture.yaml`
(`lobes doctor` checks this). Memory note: native context is 256K; the KV cache
at that length is large, so
keep `VLLM_MAX_MODEL_LEN=32768` for a first load and raise only with headroom.

## Caveats — validated during the load-test

1. **SGLang is the card's blessed runtime** (recommends `sglang serve
   --tool-call-parser qwen3_coder`). → **Resolved:** it nonetheless loads and
   serves under our vLLM image with no special flags (`trust_remote_code=False`).
   For **OpenAI tool/function calling** this model emits the Qwen3-Coder XML
   format (`<function=finish><parameter=summary>…</parameter></function>`), which
   the default `hermes` parser cannot parse (HTTP 200 but empty `tool_calls`).
   It must be served with `--tool-call-parser=qwen3_coder` — which `lobes switch`
   now **auto-selects** for this model (`lobes switch mmangkad/Qwen3.6-27B-NVFP4
   --apply` sets `VLLM_TOOL_CALL_PARSER=qwen3_coder`; override with
   `--tool-call-parser`). Verified live on `:8001`, 2026-05-27 — the
   probe returns a `finish` tool call (see
   [issue #9](https://github.com/agentculture/lobes-cli/issues/9)).
2. **`ForConditionalGeneration` + multimodal RoPE / ViT encoder.** → **Resolved
   for text:** vLLM initializes the ViT encoder but does not demand an
   image/processor path for text chat; both correctness probes passed.
3. **ModelOpt dev/rc producer** (`0.42.0rc1.dev107`). → **Resolved:** vLLM logs
   `Detected ModelOpt NVFP4 checkpoint` and the quant config parses (flagged
   "experimental format" by vLLM, but functional).
4. **New — experimental Mamba prefix caching.** With `--enable-prefix-caching`,
   vLLM sets `Mamba cache mode 'align'` and warns that prefix caching for Mamba
   layers is experimental. Functional here; worth watching for correctness drift.

## Benchmark — 2026-05-27, DGX Spark (GB10)

Image `nvcr.io/nvidia/vllm:26.04-py3`, engine `0.19.0+...nv26.04`. Served on
`:8001` via `lobes assess` / `lobes benchmark`. Engine init (download cached)
~159 s; KV cache 38.55 GiB allocated.

| Property | Value |
|---|---|
| Health / `max_model_len` | `/health` 200; `32768` (capped; 256K native) |
| Correctness | `17 × 23 = 391` ✅ (finish=stop, 389 tok); train 14:45→17:10 = 145 min ✅ (finish=stop, 1517 tok) |
| Reasoning trace field | `reasoning` (4,356-char trace) |
| **Decode throughput** | **7.9–8.0 tok/s** (batch=1, greedy, 512 tokens forced) |
| Prefill | 2,015 prompt tokens + 16 gen in 3.19 s |
| GPU memory reserved | ~70 GB (71,723 MiB) at `gpu-memory-utilization=0.6` |
| Weights on disk | ~29 GB |

### Comparison — 32B baseline (2026-05-27, GB10)

| | 27B (this model) | 32B (`nvidia/Qwen3-32B-NVFP4`) |
|---|---|---|
| Decode (batch=1) | **7.9–8.0 tok/s** | **9.7 tok/s** |
| Prefill (~2K tokens) | ~3.2 s incl. 16 gen | ~2.4 s incl. 16 gen |
| GPU reserved (util 0.6) | ~70 GB | ~72 GB |
| Weights | ~29 GB | ~20 GB |
| Native context | 256K | 32K (→131K YaRN) |
| Shape | hybrid Mamba/linear-attn + ViT (multimodal) | dense |

## Recommendation

**The 27B is the default primary (since 0.10.0)** — it is the model the consumer
agent (convertible) runs as its parent, and it brings a **much larger native
context** (256K vs the 32B's 32K/131K-YaRN) plus a **multimodal/vision** path. The
trade-off is decode speed: on this GB10 the 27B is **slower** (~8 vs ~9.7 tok/s)
despite being smaller, and it is a heavier, more-experimental path (hybrid Mamba
layers with experimental prefix caching, plus a ViT encoder unused for text).

**`nvidia/Qwen3-32B-NVFP4` remains the speed-optimised candidate** — swap it in via
`PRIMARY_MODEL` / `lobes switch` when raw text decode throughput matters more than
context length or vision. Re-run `lobes assess` / `lobes benchmark` after any vLLM
image bump — the Mamba/NVFP4 paths are young and likely to get faster.

**For MTP (speculative decoding) on the 27B,** the baseline NVFP4 export here drops
the MTP draft head (~0 % acceptance). The MTP-grafted, text-only re-export
`sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`
([`qwen3.6-27b-text-nvfp4-mtp.md`](qwen3.6-27b-text-nvfp4-mtp.md)) restores it for
vLLM speculative decoding — a candidate to benchmark against this baseline
([issue #26](https://github.com/agentculture/lobes-cli/issues/26)).
