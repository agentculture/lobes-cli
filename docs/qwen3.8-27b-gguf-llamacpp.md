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

Served by `ghcr.io/ggml-org/llama.cpp:server-cuda` via `llama-server`, which
speaks the OpenAI-compatible API. The catalog entry leaves every vLLM-only
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

## `lobes switch` and this gear

`lobes switch` configures the **vLLM** lane. Pointing it at this gear prints an
engine notice and, under `--apply`, writes `.env` **without restarting the
container** — the same protection every "needs a compose edit" gear gets, so a
healthy deployment is never taken down by a switch that could not have worked.
The llama.cpp lane itself is rendered from repo data (compose template + machine
profile + deployment shape) by the plan's tasks t4/t5, not from `VLLM_*` keys.
