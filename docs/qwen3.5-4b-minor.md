# Qwen/Qwen3.5-4B — "minor" small-brain gear

**Role:** `minor` — the fleet's 4B small-brain companion to the 27B primary.  
**Status:** `configured` (not yet load-tested on the DGX Spark).

## What it is

Qwen3.5-4B is a hybrid linear-attention + ViT multimodal checkpoint in bf16.
It is the first **unsloth-LoRA fine-tune target** in the fleet — the base model
from which quantized LoRA adapters will be derived and hot-swapped without a
full re-export.

The `role_hint` is `minor`: a lightweight inference sidekick for tasks that do
not need the primary's 27B capacity (classification, short-form generation,
tool triage, embedding-adjacent inference chains).

## Serving notes

### Text-only serving

The checkpoint ships a ViT vision tower. Until vision is tested and enabled,
serve text-only:

```yaml
command:
  - --language-model-only
```

### bf16 / unquantized — quantization="none" convention

The catalog marks this gear with `quantization="none"`, the bf16/unquantized
sentinel. `lobes switch` does **not** write `VLLM_QUANTIZATION` for a `none`
gear. This matters because the single-model template hardcodes:

```yaml
- --quantization=${VLLM_QUANTIZATION:-modelopt}
```

When `VLLM_QUANTIZATION` is absent, the default `modelopt` is applied — which
would silently corrupt a bf16 checkpoint. **You must REMOVE the `--quantization`
line from the compose `command:` by hand** before serving this model.

`lobes switch Qwen/Qwen3.5-4B` emits a compose-edit NOTE reminding you to do
this. The `--apply` path (without `--force`) blocks the container restart until
the compose file is edited, so a healthy deployment cannot be taken down by an
incompatible compose file.

### MTP / speculative decoding

This checkpoint does not carry an MTP draft head in v1. No
`--speculative-config` is set. The standard non-MTP compose-edit notice also
fires when switching from the 27B MTP primary — remove the four MTP-specific
`command:` items along with the `--quantization` line.

## Switch example

```bash
# Dry run — shows the plan and the compose-edit notices:
lobes switch Qwen/Qwen3.5-4B --machine spark

# After editing docker-compose.yml (remove --quantization + MTP lines):
lobes switch Qwen/Qwen3.5-4B --machine spark --apply --force --no-probe
```
