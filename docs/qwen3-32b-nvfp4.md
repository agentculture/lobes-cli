# Runtime model: `nvidia/Qwen3-32B-NVFP4`

The **current runtime model** model-gear runs, served by local vLLM over the
`acp` backend (the lepenseur agent consumes it). Declared in `culture.yaml` as
`vllm-local/nvidia/Qwen3-32B-NVFP4` and stood up by the packaged compose template.

## What it is

- 32B **dense** reasoning model, quantized to **NVFP4** (NVIDIA ModelOpt FP4).
- Native **32K** context (`max_model_len 32768`), extendable to ~131K via YaRN
  rope scaling.
- **Thinking mode:** emits a `<think>` reasoning trace before its answer — which
  is why it suits a deep thinker.
- Repo is **public** (no `HF_TOKEN` needed). ~20 GB on disk.

## How to run

```bash
model init --apply            # scaffold ~/.model-gear; set HF_TOKEN in .env for gated repos
docker login nvcr.io          # NGC API key, to pull the vLLM image
model serve --apply           # first run downloads ~20 GB of weights
model status                  # reports until /health is up
```

Verify:

```bash
curl -fsS http://localhost:8000/health
curl -s http://localhost:8000/v1/models   # lists nvidia/Qwen3-32B-NVFP4
```

Relevant compose flags: `--quantization=modelopt_fp4`, `--kv-cache-dtype=fp8`,
`--reasoning-parser=qwen3`, `--enable-auto-tool-choice`,
`--tool-call-parser=hermes` (OpenAI tool/function calling),
`--enable-prefix-caching`, `--gpu-memory-utilization=0.6`. Tunables in the
deployment `.env`
(`VLLM_GPU_MEM_UTIL`, `VLLM_MAX_MODEL_LEN`, `HF_CACHE`, …); `model switch`
rewrites them.

## Reading the reasoning trace

> **Field name caveat.** On the `nvcr.io/nvidia/vllm:26.04-py3` build (engine
> `0.19.0+...nv26.04`), `--reasoning-parser=qwen3` returns the `<think>` trace in
> the message field **`reasoning`**, *not* `reasoning_content`. Clients (and any
> acp `vllm-local` wiring) should read `message.reasoning`. Older vLLM builds use
> `reasoning_content` — check the field name against your image.

A reasoning model spends most of its tokens thinking, so give it room: a tight
`max_tokens` can be consumed entirely inside the trace, leaving `content` empty
with `finish_reason: length`.

## Live test — 2026-05-27, DGX Spark (GB10)

Verified end-to-end on the GB10 (121 GB unified memory). Served on port `8001`
during the test (compose default is `8000`) to avoid a co-resident service.

| Property | Value |
|---|---|
| Image / engine | `nvcr.io/nvidia/vllm:26.04-py3` / vLLM `0.19.0+...nv26.04` |
| Weights on disk | ~20 GB |
| GPU memory reserved | ~72 GB (`gpu-memory-utilization=0.6`; 74,136 MiB observed) |
| Health / models | `/health` 200; `/v1/models` lists the model, `max_model_len 32768` |
| Correctness | "14:45→17:10 = 145 min" ✅; "17 × 23 = 391" ✅, both with full reasoning trace |
| **Decode throughput** | **~9.7 tok/s** (batch=1, greedy, 512 tokens forced; identical across 2 runs) |
| **Prefill** | ~2,014 prompt tokens in ~0.7 s (~2,800 tok/s); 2.37 s incl. 16 decode tokens |

Decode at ~10 tok/s reflects a 32B dense model on the GB10's low-power unified
memory — adequate for a deliberate, write-by-thinking agent, not for
high-throughput serving.

### Known noise

`env_file: .env` passes the compose-interpolation vars (`VLLM_MODEL`,
`VLLM_PORT`, …) into the container, so vLLM logs harmless
`Unknown vLLM environment variable detected: VLLM_*` warnings at startup. They do
not affect serving.

## Fallback

If a vLLM build rejects the `nvidia/` ModelOpt checkpoint, set
`VLLM_MODEL=RedHatAI/Qwen3-32B-NVFP4` and drop `--quantization` from the compose
`command` (the RedHatAI checkpoint is vLLM-native).
