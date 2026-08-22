# unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M — the llama.cpp gear

**Status: `configured`, NOT validated.** The facts below are read off the
downloaded GGUF file and off `llama-server`'s own flag surface. **No acceptance
transcript exists**, so nothing here — and nothing anywhere else in the repo —
may claim this lane VALIDATED (#108). The covering plan
(`docs/plans/2026-08-22-qwen3-8-gguf-llamacpp.md`) lands the evidence in task
t10; this doc gets its measured numbers then.

This is the catalog's **first non-vLLM gear**. It exists to record the engine
axis (`lobes/catalog.py`'s `engine` field, plan task t3) against a real
checkpoint, not to re-point any role: `role_hint` is `candidate`, so no tier
alias resolves to it and the fleet's `cortex` primary is unchanged.

## The checkpoint

Measured off the downloaded file, 2026-08-23, on the Jetson AGX Orin 64 GB this
gear is aimed at:

| fact | value |
|---|---|
| `general.architecture` | `qwen35` |
| parameters | 27.32B |
| file size | 15.32 GiB |
| quantization | `Q4_K_M` (Unsloth Dynamic 2.0, `UD-Q4_K_M`) |
| native context | 262144 (256K), the GGUF's own declared ceiling |

It is the same upstream checkpoint the fleet's vLLM `cortex` primary serves
(`unsloth/Qwen3.8-27B-NVFP4` — see `docs/qwen3.8-27b-nvfp4.md`), in a different
container format for a different engine.

## The engine — why the vLLM fields are empty

Served by `llama-server`, which speaks the OpenAI-compatible API, from the
digest-pinned image the lane carries (see "The lane" below). The catalog entry leaves every vLLM-only
field empty **on purpose**:

- **`tool_parser=""`** — `llama-server` has no `--tool-call-parser` and no
  `--reasoning-parser`. It takes `--jinja` and uses the model's own embedded
  chat template, auto-detecting the tool-call and reasoning shapes
  (`--reasoning-format auto`, `--chat-template-kwargs`). `infer_parser` answers
  `qwen3_coder` for any `qwen3.8` id; that answer names a **vLLM flag that does
  not exist here**, so it is deliberately not carried. `lobes switch` skips it
  too (`serves_with_vllm`).
- **`quantization=""`** — `Q4_K_M` lives inside the `.gguf` file. There is no
  flag to pass.
- **`speculative_config=""`** — llama.cpp **ignores** the checkpoint's MTP head.
  The NVFP4 sibling's self-hosted MTP draft does not run on this engine, so this
  gear has **no speculative decoding**. Declaring the sibling's config would
  advertise a capability the lane cannot serve.
- **`moe_backend=""` / `hf_overrides=""`** — vLLM serve flags; llama.cpp has
  neither surface.

## Feature gaps versus the vLLM cortex

Recorded here so no caller infers parity from the shared checkpoint id. Each is
a **declared** gap from the engine's flag surface, to be re-checked against the
plan's live transcript:

| feature | vLLM cortex | this lane |
|---|---|---|
| MTP speculative decoding | self-hosted draft, measured | **ignored by the engine** |
| ViT (image / video intake) | yes | not served on this lane |
| `preserve_thinking` (#93) | `--default-chat-template-kwargs` | UNVERIFIED |
| strict tools / xgrammar (colleague#320) | armed | UNVERIFIED |


## The lane (plan t4)

`llamacpp-primary` in `lobes/templates/fleet/docker-compose.yml` — the fleet's
first non-vLLM generate service. Properties worth knowing before you touch it:

| property | value | why |
|---|---|---|
| image | `ghcr.io/nvidia-ai-iot/llama_cpp@sha256:f7c67c10…` | Pinned by digest, never a tag. llama.cpp build `38406d597` (10373) on CUDA 13, arm64, `CUDAARCHS=87;110`. |
| gating | `profiles: [llamacpp]` | Opt-in like `vllm-muse`/`vllm-worker`/`vllm-minor` — no existing deployment starts it. |
| exposure | `expose: "8000"`, **no `ports:`** | Reachable only on the compose network, matching the `vllm-multimodal` precedent. The gateway origin is the only way in. |
| GPU request | the shipped `deploy.resources` stanza | A csv-mode board rewrites it to `runtime: nvidia` through its card profile's `gpu_access = "runtime"` declaration and the generated `docker-compose.gpu.yml` — see below. |
| model | `/models/${LLAMACPP_GGUF}`, mounted **read-only** | The lane never downloads. The served file is the one the operator placed. |
| health | `curl -f http://localhost:8000/health` | `llama-server` answers `{"status":"ok"}` once loaded; `start_period` covers the measured 84 s cold load. |

**The image was chosen by measurement, 2026-08-23 on the Jetson AGX Orin 64GB**
(`llama-bench -p 512 -n 128 -ngl 99`):

| build | CUDA | flash attn | pp512 | tg128 |
|---|---|---|---|---|
| `ghcr.io/nvidia-ai-iot/llama_cpp` 10373 | 13 | on | **64.79** | **2.61** |
| `ghcr.io/nvidia-ai-iot/llama_cpp` 10373 | 13 | off | 63.56 | 2.57 |
| `ghcr.io/ggml-org/llama.cpp` 10573 | 12 | on | 63.10 | 2.53 |
| `ghcr.io/ggml-org/llama.cpp` 10573 | 12 | off | 61.98 | 2.52 |

The ggml-org image is the recorded **runner-up**, not a drop-in: swapping to it
is a deliberate act that should be re-measured.

**Serve flags** are the measured-best set: `-ngl 99 -c 262144 --jinja -np 1
-fa on`. They are written as separate `--flag value` list items because
llama.cpp's own arg parser **rejects** the `--flag=value` form every vLLM lane
uses (verified against this image: `error: invalid argument: --ctx-size=4096`).

**No vLLM flag appears in it** — no `--quantization`, no
`--gpu-memory-utilization`, no `--tool-call-parser`, no `--reasoning-parser`,
no `--speculative-config`. Those surfaces do not exist on this engine, and a
translated-looking flag would be a claim the lane cannot honour.

## Rendering it (plan t5)

Nothing about the lane is hand-configured. The `orin` card profile declares
`cortex` on this gear, and the engine follows from the catalog entry for that
model id:

```bash
lobes init --profile orin --shape orin-cortex --apply
```

renders, from repo data alone:

* `.env` — `PRIMARY_MODEL` / `PRIMARY_SERVED_NAME` = this gear,
  `PRIMARY_MAX_MODEL_LEN=262144`, `PRIMARY_URL=http://llamacpp-primary:8000`,
  `COMPOSE_PROFILES=llamacpp`, `MULTIMODAL_FEASIBLE=false`. No
  `PRIMARY_GPU_MEM_UTIL` — this engine has no such flag, so declaring one would
  be a dead knob;
* `docker-compose.shape.yml` — parks `vllm-primary` **and** `vllm-multimodal` in
  the inert `shape-dropped` compose profile. Parking the vLLM cortex lane while
  cortex is *hosted* is the engine swap: both lanes running at once on a
  61.3 GiB board is the failure that prevents;
* `docker-compose.gpu.yml` — `!reset`s every GPU service's `deploy:` stanza and
  sets `runtime: nvidia`, because this board's NVIDIA container toolkit resolves
  to legacy **csv** mode and refuses the `deploy.resources` form at container
  create. `llamacpp-primary` is in that list.

The caller-facing contract does not move: `model=cortex|main|hard` still routes
to `cortex`, and the gateway needed no code change to reach a different engine.

## Memory, measured

At the full 262144 window on the Orin (2026-08-23):

| term | value |
|---|---|
| weights | 15.33 GiB |
| KV cache | 16.00 GiB (**64 KiB/token**) |
| resident total | **~33 GiB** (predicted 31.33 from the GGUF header) |
| load time | 84 s cold, 15.6 s warm-cache |

KV is cheap because only **16 of the 65 layers** hold a per-token cache
(`full_attention_interval = 4`), and those use GQA at `head_count_kv = 4`,
`key_length = value_length = 256`. The other 49 are Mamba/SSM with constant
state. `-c` is the only memory dial this engine has.

That footprint is why `orin-cortex` drops `senses`: ~33 GiB plus senses' ~27.6
GiB does not fit in 61.3 GiB, and the board has **zero swap**.

## `lobes switch` and this gear

`lobes switch` configures the **vLLM** lane. Pointing it at this gear prints an
engine notice and, under `--apply`, writes `.env` **without restarting the
container** — the same protection every "needs a compose edit" gear gets, so a
healthy deployment is never taken down by a switch that could not have worked.
The llama.cpp lane itself is rendered from repo data (compose template + machine
profile + deployment shape) by the plan's tasks t4/t5, not from `VLLM_*` keys.
