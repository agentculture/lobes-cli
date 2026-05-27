# Candidate model: `mmangkad/Qwen3.6-27B-NVFP4`

A candidate alternative runtime model. **Load-tested live on DGX Spark (GB10),
2026-05-27** — it loads and serves cleanly under the vLLM image lepenseur
already runs. Tracked by [issue #6](https://github.com/agentculture/lepenseur/issues/6).

Source: <https://huggingface.co/mmangkad/Qwen3.6-27B-NVFP4> — public, Apache-2.0.

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
$ docker exec lepenseur-vllm python3 -c \
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
.claude/skills/model-runner/scripts/model-runner.sh \
  switch mmangkad/Qwen3.6-27B-NVFP4 --port 8001 --max-model-len 32768
# (or edit .env: VLLM_MODEL / VLLM_SERVED_NAME / VLLM_PORT, then docker compose up -d)
```

`VLLM_SERVED_NAME` must match the part after `vllm-local/` in `culture.yaml`.
Memory note: native context is 256K; the KV cache at that length is large, so
keep `VLLM_MAX_MODEL_LEN=32768` for a first load and raise only with headroom.

## Caveats — validated during the load-test

1. **SGLang is the card's blessed runtime** (recommends `sglang serve
   --tool-call-parser qwen3_coder`). → **Resolved:** it nonetheless loads and
   serves under our vLLM image with no special flags (`trust_remote_code=False`).
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
`:8001` via `model-runner assess`. Engine init (download cached) ~159 s; KV
cache 38.55 GiB allocated.

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

**Keep `nvidia/Qwen3-32B-NVFP4`.** The 27B loads cleanly and answers correctly,
but on this GB10 it is **slower on decode** (~8 vs ~9.7 tok/s) despite being
smaller, and it is a heavier, more-experimental path (hybrid Mamba layers with
experimental prefix caching, plus an unused ViT encoder). For a text-only deep
thinker, that trade does not pay off today.

Switch only if a concrete need appears that the 32B cannot meet: a **much larger
context** (256K native vs 32K/131K-YaRN) or **multimodal/vision** input. Re-run
`model-runner assess` after any vLLM image bump — the Mamba/NVFP4 paths are young
and likely to get faster.
