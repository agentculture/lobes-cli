# Fallback model: `RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4`

The **dense fallback** the gateway fleet pairs with the default primary
(`mmangkad/Qwen3.6-27B-NVFP4`). It **replaces the Qwen3.6-35B-A3B MoE**, which
never loaded on this GB10 (OOM co-resident, stall solo — see
[`docs/qwen3.6-35b-a3b-nvfp4.md`](qwen3.6-35b-a3b-nvfp4.md)). Mistral is dense,
loads reliably, and serves text + tool calls. See
[`docs/gateway-fleet.md`](gateway-fleet.md) for the fleet topology.

Source: <https://huggingface.co/RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4>.

> **Status: load-tested 2026-05-30 on the DGX Spark (GB10).** Loaded solo on port
> `8001`, reached `/health`, passed the arithmetic + tool-calling probes, and
> benchmarked at **~14.9 tok/s** decode. The serving recipe below is the one that
> actually works on the `nvcr.io/nvidia/vllm:26.04-py3` build — two non-obvious
> flags are required (see "Serving gotchas").

## What it is

- **24B dense** instruct model, quantized to **NVFP4** (vLLM picks the
  `NvFp4LinearBackend.FLASHINFER_CUTLASS` GEMM). The checkpoint is
  **compressed-tensors** format (RedHatAI / llm-compressor), so vLLM auto-detects
  the quantization — it is *not* the ModelOpt FP4 the `nvidia/` + `mmangkad/`
  checkpoints use.
- **Vision-capable** (`Mistral3ForConditionalGeneration`, a Pixtral-style
  encoder), but the fleet serves it **text-only** (images limited to 0 — see
  below). Tool calling works; vision is unused by the fallback.
- **Instruct, not a thinker:** no `<think>` reasoning trace (it reasons inline in
  `content`). Served **without** `--reasoning-parser`.
- **128K** native context (`max_model_len` capped to **32K** for the first load).
- Repo is **public** (no `HF_TOKEN`). ~16 GiB on disk (4 safetensors shards).

## How it runs in the fleet

Configured via the `FALLBACK_*` keys in the fleet `.env` (scaffolded by
`model init --fleet`); served by the `model-gear-vllm-fallback` container:

```dotenv
FALLBACK_MODEL=RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4
FALLBACK_SERVED_NAME=RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4
FALLBACK_MAX_MODEL_LEN=32768
FALLBACK_GPU_MEM_UTIL=0.35
FALLBACK_TOOL_CALL_PARSER=mistral
FALLBACK_QUANTIZATION=compressed-tensors
```

The fleet compose hard-codes the two Mistral-specific flags (`--tokenizer-mode
mistral` and `--limit-mm-per-prompt {"image":0}`) on the fallback service. Address
it through the gateway by name:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -d '{"model":"RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4","messages":[{"role":"user","content":"hi"}]}'
```

## Serving gotchas (the recipe that works)

The working `vllm serve` flags on the `nv26.04` build are:

```text
--tokenizer-mode=mistral
--limit-mm-per-prompt={"image":0}
--enable-auto-tool-choice  --tool-call-parser=mistral
--kv-cache-dtype=fp8  --max-model-len=32768  --enable-prefix-caching
# NVFP4 is compressed-tensors → auto-detected (no --quantization needed;
# pass --quantization=compressed-tensors to be explicit). No --reasoning-parser.
```

Two flags are non-obvious and were found the hard way:

1. **Tool calling needs the *mistral* tokenizer.** Served with the HF tokenizer
   (`--tokenizer-mode auto`, the default), the model loads and answers text fine,
   but a `tool_choice:"auto"` request returns the raw markup
   `[TOOL_CALLS]get_weather[ARGS]{"city":"Paris"}` in `content` with an **empty
   `tool_calls`** — the `mistral` parser doesn't recognise the HF chat template's
   format. The **mistral tokenizer** (the checkpoint ships `tekken.json`) makes
   the parser produce a proper `tool_calls` array (`finish_reason: tool_calls`,
   `content: null`).
2. **The mistral tokenizer alone crashes the multimodal profiler.** With
   `--tokenizer-mode mistral` and no image limit, engine init dies in the Pixtral
   dummy-input path:
   `AssertionError: Expected to decode 1 token, got 3` (the mistral_common
   tokenizer vs. the Pixtral image token). **`--limit-mm-per-prompt {"image":0}`**
   disables image inputs and sidesteps the crash — acceptable because the fallback
   is text + tools only.

The full mistral stack (`--config-format mistral --load-format mistral`) is **not**
an option: this NVFP4 repo ships HF-format `config.json` + sharded `safetensors`
(no `params.json`/`consolidated.safetensors`), so only the *tokenizer* is loaded
in mistral mode.

> **Running it as a standalone single model.** Mistral's supported path is the
> fleet fallback (the fleet compose encodes the recipe above). `model switch
> RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4` sets the catalog quantization
> (`compressed-tensors`) and the `mistral` tool parser, but the single-model
> compose targets the dense/Qwen primaries and does **not** add
> `--tokenizer-mode mistral` / `--limit-mm-per-prompt {"image":0}` — add those to
> the compose `command` by hand if you serve Mistral standalone, or tool calls
> will leak as text.

## Benchmark — 2026-05-30, DGX Spark (GB10)

Loaded **solo** on port `8001` (the 27B primary stopped first to free memory),
`gpu-memory-utilization=0.4`.

| Property | Value |
|---|---|
| Image / engine | `nvcr.io/nvidia/vllm:26.04-py3` / vLLM `0.19.0+...nv26.04` |
| Weights on disk | ~16 GiB (4 safetensors shards) |
| Model loading | 15.05 GiB, weights loaded in ~83 s |
| KV cache | 30.69 GiB available; max concurrency **12.28x** at 32K tokens |
| GPU memory (EngineCore) | ~49,557 MiB (`gpu-memory-utilization=0.4`) |
| Health / models | `/health` 200; `/v1/models` lists the model, `max_model_len 32768` |
| Correctness | `17 × 23 = 391` ✅; `14:45→17:10 = 145 min` computed correctly (see note) |
| Reasoning trace | none (instruct model — answers inline in `content`) |
| Tool calling (`tool_choice:auto`) | ✅ proper `tool_calls` array (`finish_reason: tool_calls`) |
| **Decode throughput** | **14.9 / 14.9 tok/s** (batch=1, greedy, 512 tokens forced; identical across 2 runs) |
| **Prefill** | 2,009 prompt tokens + 16 gen in **1.49 s** (~1,350 tok/s) |

Decode at ~15 tok/s is **~50 % faster than the dense 32B** (`~9.7 tok/s`) — a
smaller 24B reads fewer bytes per token — and it actually *loads*, unlike the 35B
MoE. Suitable as the always-warm fallback the gateway fails over to.

> **Note on the time-duration probe.** `model assess` marks
> `14:45→17:10 = 145 min` as FAIL because its strict `"145" in content` check
> didn't match that one greedy run's verbose output. Manual re-runs of the exact
> prompt return the correct answer (the model derives `145` and boxes it). Treat
> the FAIL as a checker-strictness / fp8-KV-nondeterminism artifact, not a wrong
> answer.

### Known noise

`env_file: .env` passes the compose-interpolation vars into the container, so
vLLM logs harmless `Unknown vLLM environment variable detected: VLLM_*` warnings
at startup. They do not affect serving.
