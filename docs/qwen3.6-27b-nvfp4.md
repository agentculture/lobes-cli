# Candidate model: `mmangkad/Qwen3.6-27B-NVFP4`

A candidate alternative runtime model. **Architecturally supported** by the
vLLM image lepenseur already runs, but **not yet load-tested live** — see the
benchmark plan below. Tracked by [issue #6](https://github.com/agentculture/lepenseur/issues/6).

Source: <https://huggingface.co/mmangkad/Qwen3.6-27B-NVFP4> — public, Apache-2.0.

## What it is

- NVFP4 (NVIDIA ModelOpt) quantization of **`Qwen/Qwen3.6-27B`**.
- `config.json`: `architectures: ["Qwen3_5ForConditionalGeneration"]`,
  `model_type: qwen3_5`, 64 layers, `hidden_size 5120`,
  `max_position_embeddings 262144` (**256K** context), multimodal RoPE
  (`mrope_interleaved`, `mrope_section`).
- ~20B effective params after compression; ~20 GB on disk (BF16 / F8_E4M3 / U8
  tensors). ModelOpt producer `0.42.0rc1.dev107` (a dev/rc build).

## Is it supported here? — Yes (architecture), pending live load

The deciding check: query the **running** vLLM engine's model registry rather
than guess.

```text
$ docker exec lepenseur-vllm python3 -c \
  "from vllm.model_executor.models.registry import ModelRegistry; \
   print('Qwen3_5ForConditionalGeneration' in ModelRegistry.get_supported_archs())"
True
```

The `nvcr.io/nvidia/vllm:26.04-py3` image (engine `0.19.0+...nv26.04`) registers
`Qwen3_5ForConditionalGeneration` (plus `Qwen3_5MoeForConditionalGeneration` and
`Qwen3_5MTP`) — the exact architecture this checkpoint declares. The quant flag
(`--quantization=modelopt_fp4`) and `--reasoning-parser=qwen3` are the same ones
already working for the 32B. So the same compose can serve it.

"Registered" means vLLM can instantiate the model class; it does not prove the
weights load and serve cleanly. That requires the live load-test below.

## How to run (same compose, model override)

```bash
# in .env
VLLM_MODEL=mmangkad/Qwen3.6-27B-NVFP4
VLLM_SERVED_NAME=mmangkad/Qwen3.6-27B-NVFP4   # must match culture.yaml's vllm-local/<name>
# keep --quantization=modelopt_fp4 and --reasoning-parser=qwen3 (already in compose)
docker compose up -d
```

Memory note: at 256K context the KV cache is large. Keep
`VLLM_MAX_MODEL_LEN=32768` (or similar) for a first load; only raise it with
headroom to spare. The GB10 has 121 GB unified memory total.

## Caveats to validate during the load-test

1. **SGLang is the blessed runtime.** The model card recommends `sglang serve`
   (with `--tool-call-parser qwen3_coder`), not vLLM. vLLM support is present in
   the registry but is not the card's documented path.
2. **`ForConditionalGeneration` + multimodal RoPE.** The arch and `mrope` config
   suggest a vision/multimodal lineage; text-only chat should still serve, but
   confirm vLLM does not demand an image/processor path at load.
3. **ModelOpt dev/rc producer** (`0.42.0rc1.dev107`) — verify the quant config
   parses under this vLLM build.

## Benchmark plan (to be filled when load-tested)

Run the same methodology used for the 32B (see
[`qwen3-32b-nvfp4.md`](qwen3-32b-nvfp4.md)) so the two are comparable:

- Health + `/v1/models` reachable.
- Correctness on the same two probes (train-times, `17 × 23`), confirming the
  `reasoning` field populates.
- Decode throughput: 512 tokens forced (`ignore_eos`), batch=1, greedy.
- Prefill: ~2K-token prompt, 16-token gen.
- Record image/engine version, weights-on-disk, and GPU memory reserved.

| Property | Value |
|---|---|
| Decode throughput | _TBD_ |
| Prefill | _TBD_ |
| GPU memory reserved | _TBD_ |
| Correctness | _TBD_ |

### For comparison — 32B baseline (2026-05-27, GB10)

~9.7 tok/s decode (batch=1), ~2,800 tok/s prefill, ~72 GB reserved at
`gpu-memory-utilization=0.6`.

## Recommendation

**Pending the load-test.** Until the live numbers above exist, **keep
`nvidia/Qwen3-32B-NVFP4`** as the runtime model — it is verified end-to-end on
this hardware. Revisit a switch only if the 27B load-test (issue #6) shows it
loads cleanly under vLLM *and* offers a worthwhile trade (faster decode at ~20B,
larger usable context) without the multimodal/SGLang caveats biting.
