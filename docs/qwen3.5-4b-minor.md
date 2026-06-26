# Qwen/Qwen3.5-4B — "minor" small-brain gear

**Role:** `minor` — the fleet's 4B small-brain companion to the 27B primary.  
**Status:** `configured` (not yet load-tested on the DGX Spark).

## What it is

Qwen3.5-4B is a hybrid linear-attention + ViT multimodal checkpoint in bf16.
The `role_hint` in the catalog is `minor`: a lightweight inference sidekick for
tasks that do not need the primary's 27B capacity (classification, short-form
generation, tool triage, routing decisions).

The checkpoint ships a ViT vision tower; the fleet serves it **text-only** via
`--language-model-only` — the tower is dropped at load time, leaving only the
language trunk (no vision capability in this configuration).

It is the fleet's first **unsloth-LoRA fine-tune target**: the bf16 base model
from which quantized LoRA adapters will be derived and hot-swapped without a
full re-export. **Fine-tuning is deferred** — the model is catalogued and
co-residency is wired, but no adapter training has been performed yet.

## Verbs

Three read-only verbs target the minor lobe directly. None of them writes
`.env`, `docker-compose.yml`, or any other file; no `--apply` flag is needed
or accepted.

### `lobes run minor`

Send a prompt to the minor lobe and print the reply.

```bash
lobes run minor "Summarize this commit message in one sentence: feat: add minor lobe co-residency"
```

Options: `--system "..."` (prepend a system message), `--max-tokens N`,
`--json` (emit the full chat-completion JSON object instead of just the text),
`--base-url URL` (default: `http://localhost:8000/v1`), `--model <id>` (override
the catalog lookup by `role_hint="minor"`).

### `lobes route`

Ask the minor lobe to classify a task description into the most appropriate
catalog gear role. Governance is overlaid via `lobes.minor.governance.decide`:
any recognised escalation condition forces `escalate=true` in the output
regardless of the model's suggestion. Agents **must** honour the escalation flag.

```bash
lobes route "Summarize this PR description for a release note"
```

Output (plain text by default, `--json` for structured):

```text
chosen_gear: minor
escalate:    no
confidence:  0.92
reason:      Quick summarization task, well within minor lobe capacity.
```

Structured output fields: `chosen_gear` (catalog gear role), `escalate` (bool),
`confidence` (float 0–1, self-reported and clamped), `reason` (one-sentence
explanation). Routing targets are catalog gear roles only — not tools or mesh
agents.

### `lobes eval minor`

Run a JSONL eval suite against the minor lobe (or any OpenAI-compatible
endpoint). Each suite line is a JSON case object with a `prompt` and exactly one
expectation field (`expect_substring` or `expect_regex`). Blank lines and lines
starting with `#` are skipped.

```bash
lobes eval minor --suite tests/fixtures/minor_suite.jsonl
```

Reports per-case `PASS` / `FAIL` and an aggregate `passed/total`. Pass `--json`
for a structured report (`passed`, `total`, `cases`). Exit code is always 0 —
pass/fail lives in the report, not the exit code. A missing suite file is the
only non-zero exit. All three verbs default `--base-url` to the gateway
`http://localhost:8000/v1` (override with `--base-url`).

## Governance

Governance is role-keyed (`role_hint == "minor"`), not model-keyed — swapping
the underlying model in the catalog does not touch this policy.

### Allowed duties

The minor lobe may perform these duties locally without escalation:

- `prepare`
- `classify`
- `format`
- `validate`
- `suggest`
- `summarize`
- `route`

### Forbidden actions

The minor lobe must **never** perform these actions; they always escalate to the
primary lobe or a human reviewer, regardless of any other conditions:

- `approve`
- `finalize`
- `delete`
- `deploy`
- `architectural_decision`

### Escalation conditions

Any **single** matching condition forces escalation, even when the duty is
otherwise allowed:

- `needs_codebase_context`
- `security_sensitive`
- `architectural_decision`
- `write_or_delete_operation`
- `final_review_required`

The governance engine (`lobes.minor.governance.decide`) is **fail-closed**: an
unknown duty (not in allowed, not in forbidden) also escalates rather than
proceeding locally.

## Safety contract

The three minor-lobe verbs (`run`, `route`, `eval`) are **read-only** — they
make HTTP requests to the local fleet gateway and never touch the file system.
No `--apply` flag is needed or accepted for any of them.

The remaining write verbs (`switch`, `serve`, `stop`, `init`, `tunnel`) are
**dry-run by default** and require `--apply` to commit any change. This
safe-by-default contract is mandatory: agents call CLIs in loops, and a
destructive default is a bug.

## Warm co-residency (opt-in)

The fleet compose ships an opt-in `vllm-minor` service under `profiles: [minor]`.
By default the fleet runs unchanged (primary + embed + rerank); the minor backend
is dormant.

### Activating the minor service

```bash
# Option 1 — add to .env:
COMPOSE_PROFILES=minor

# Option 2 — pass the profile on the command line:
docker compose --profile minor up -d
```

### Gateway env gate

The gateway routes requests for `model: Qwen/Qwen3.5-4B` to the minor backend
**only** when both variables are set in the gateway's environment:

```text
MINOR_BASE_URL=http://vllm-minor:8000
MINOR_SERVED_NAME=Qwen/Qwen3.5-4B
```

When these are empty (the compose default), the gateway ignores the minor backend
even if `vllm-minor` is running. Set them in `.env` or pass them explicitly to
the gateway service when enabling the profile.

If the minor backend refuses the connection or returns a 5xx before any response
body, the gateway fails over to the primary.

### GPU budget

At `VLLM_MINOR_GPU_MEM_UTIL=0.10` (~13 GiB) the 4B bf16 model co-resides
alongside the 27B primary (`PRIMARY_GPU_MEM_UTIL=0.6`, ~75 GiB) on the 128 GB
GB10. Together with the two ~0.6B gears (`*_GPU_MEM_UTIL=0.06` each) the four
services leave ~35 GiB free. The context window is capped at
`VLLM_MINOR_MAX_MODEL_LEN=32768` to keep KV-cache small in the co-resident role.

This budget is part of the fleet design; the minor service is declared
`configured` — it has not been load-tested on the DGX Spark yet. Validate live
(`spark memory` / `nvidia-smi` at `lobes fleet up`) before relying on it in
production.

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

## Alternative: cosmicproc/Qwen3.5-4B-NVFP4 (untested)

A community NVFP4 export (`cosmicproc/Qwen3.5-4B-NVFP4`) exists for operators
who want a quantized minor lobe. It is **not in the lobes catalog** and has
not been tested on the DGX Spark. If you use it, the `quantization="none"`
convention does not apply — this variant needs
`--quantization=modelopt_fp4` (or `VLLM_QUANTIZATION=modelopt_fp4`). Load-test
before co-deploying alongside the primary.

## Switch example

```bash
# Dry run — shows the plan and the compose-edit notices:
lobes switch Qwen/Qwen3.5-4B --machine spark

# After editing docker-compose.yml (remove --quantization + MTP lines):
lobes switch Qwen/Qwen3.5-4B --machine spark --apply --force --no-probe
```
