# Fallback model: `mmangkad/Qwen3.6-35B-A3B-NVFP4`

The **MoE fallback** the gateway fleet pairs with the default primary
(`mmangkad/Qwen3.6-27B-NVFP4`). See [`docs/gateway-fleet.md`](gateway-fleet.md) for
the fleet topology; this doc records what the model is and how it is configured in
the fleet.

Source: <https://huggingface.co/mmangkad/Qwen3.6-35B-A3B-NVFP4>.

> **Status: configured, not yet load-tested on this hardware.** The numbers below
> are *expectations* from the architecture, not measured values. Fill in the
> Benchmark table from a live `model fleet up` → `model assess` / `model benchmark`
> run (and confirm the quantization/parser caveats) before relying on them.

## What it is

- An **NVFP4 (Mixture-of-Experts)** checkpoint: ~35B total parameters, **~3B
  active per token** (`A3B`). vLLM loads *all* experts into memory; the small
  active set only reduces per-token compute.
- Decode is memory-bandwidth bound on the GB10 (~273 GB/s shared). Reading only
  ~3B active params per token (≈1.5 GB at 4-bit) gives an **expected decode
  ceiling far above the dense 32B** (which reads ~18 GB/token) — the reason it is
  the fast fallback. *Confirm live.*

## How it runs in the fleet

Configured via the `FALLBACK_*` keys in the fleet `.env` (scaffolded by
`model init --fleet`); served by the `model-gear-vllm-fallback` container:

```dotenv
FALLBACK_MODEL=mmangkad/Qwen3.6-35B-A3B-NVFP4
FALLBACK_SERVED_NAME=mmangkad/Qwen3.6-35B-A3B-NVFP4
FALLBACK_MAX_MODEL_LEN=32768
FALLBACK_GPU_MEM_UTIL=0.30          # both models warm: keep primary+fallback well under 1.0
FALLBACK_TOOL_CALL_PARSER=qwen3_coder
FALLBACK_QUANTIZATION=modelopt_fp4
```

Address it through the gateway by name (or set `GATEWAY_ALIASES` for a short
alias):

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -d '{"model":"mmangkad/Qwen3.6-35B-A3B-NVFP4","messages":[{"role":"user","content":"hi"}]}'
```

## Caveats to confirm on first load

1. **Tool-call format.** Qwen3.6 emits the Qwen3-Coder **XML** function format, so
   the backend is served with `--tool-call-parser=qwen3_coder` (not the `hermes`
   parser the dense Qwen3-32B uses). `model_gear.runtime._parser.infer_parser`
   already maps `qwen3.6` → `qwen3_coder`. Verify a `tool_choice:"auto"` probe
   returns a `finish` tool call.
2. **Quantization format.** The fleet defaults `FALLBACK_QUANTIZATION=modelopt_fp4`
   (as for the `nvidia/` checkpoints). This community (`mmangkad`) checkpoint may
   instead be a compressed-tensors NVFP4 — if vLLM rejects `modelopt_fp4`, drop or
   change `FALLBACK_QUANTIZATION`.
3. **`--trust-remote-code`.** The fleet compose omits it (as the single-model
   template does). If this checkpoint ships custom modeling code, vLLM will say so
   on load; add it back deliberately (it lets repo code run in-container alongside
   `HF_TOKEN` and the mounted cache).
4. **Architecture support.** Confirm the engine registers the checkpoint's
   architecture, as done for the 27B sibling:
   `docker exec model-gear-vllm-fallback python3 -c "from
   vllm.model_executor.models.registry import ModelRegistry;
   print(ModelRegistry.get_supported_archs())"`.

## Benchmark — pending

Fill from a live run (`model fleet up --apply`, then `model assess` / `model
benchmark` against `:8000` with this model's name):

| Property | Value |
|---|---|
| Health / `max_model_len` | *pending* |
| Correctness (`17×23`, train 14:45→17:10) | *pending* |
| Reasoning trace field | *pending* |
| Tool calling (`tool_choice:auto`, `qwen3_coder`) | *pending* |
| **Decode throughput** | *pending* (expected ≫ the dense 32B's ~9.7 tok/s) |
| Prefill (~2K tokens) | *pending* |
| GPU memory reserved (at `FALLBACK_GPU_MEM_UTIL`) | *pending* |
| Co-resident total (primary + fallback) | *pending* — watch for OOM |
