# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.1] - 2026-05-27

### Changed

- **SonarCloud cleanup (no behavior change).** Split `cmd_switch` into
  `_select_parser` / `_emit_dry_run` / `_apply_switch` helpers to bring its
  cognitive complexity under the gate, and hoisted the repeated `"(unset)"`
  literal in `model status` into a `_UNSET` constant.

## [0.8.0] - 2026-05-27

### Added

- **Per-model tool-call parser auto-selection.** New `model_gear/runtime/_parser.py`
  `infer_parser()` maps a model name to its parser (`qwen3_coder` for
  Qwen3-Coder / Qwen3.6, `hermes` for Qwen3 dense, unknown → leave untouched).
  `model switch` now picks the right parser automatically so tool calling keeps
  working across a switch without the caller remembering it; `--tool-call-parser`
  still overrides ([issue #13](https://github.com/agentculture/model-gear/issues/13)).
- **Post-switch / post-start tool-calling probe.** `model switch --apply` and
  `model serve --apply` now probe `tool_choice:"auto"` once the container is
  healthy and report PASS/FAIL (with the called tool names) — reusing the
  existing `assess` probe. `--no-probe` skips it; the probe never aborts the
  command (unreachable / HTTP 400 degrade to a FAIL result).
- **`model status` reports the active `tool_call_parser`** (`VLLM_TOOL_CALL_PARSER`),
  so "which gear am I in" is complete without `docker inspect`.

### Changed

- **`lepenseur` is retired; the deployed agent is now `model-gear`.** The tool and
  the deployed agent share one identity. Updated `culture.yaml` (`suffix: model-gear`),
  the `AGENTS.md` system prompt, `model whoami` / `learn` / `explain` output, the
  posting nick (`.claude/skills.local.yaml.example`), the compose/`.env` templates,
  `README.md`, and `CLAUDE.md` (the former "two identities" section now describes
  one).

### Fixed

## [0.7.0] - 2026-05-27

### Added

- **OpenAI tool/function calling** on the served vLLM model. The packaged compose
  template (`model_gear/templates/docker-compose.yml`) now serves with
  `--enable-auto-tool-choice` and `--tool-call-parser=${VLLM_TOOL_CALL_PARSER:-hermes}`,
  so `tool_choice:"auto"` requests return a `tool_calls` array instead of HTTP
  400. Additive — plain chat/reasoning is unaffected, no extra GPU/memory cost.
  Unblocks coder-agent harnesses that drive the model entirely through tool calls
  ([issue #9](https://github.com/agentculture/model-gear/issues/9)).
- **`VLLM_TOOL_CALL_PARSER`** env var (default `hermes`) + **`model switch
  --tool-call-parser`** — the parser is per-model: `hermes` fits Qwen3 dense
  (e.g. `Qwen3-32B`), while Qwen3-Coder / Qwen3.6 checkpoints emit the XML
  function format and need `qwen3_coder`. `switch` writes the var only when the
  flag is given, so retuning a model never clobbers its parser.
- **`model assess --tools`** — an opt-in tool-calling probe that verifies a
  `tool_choice:"auto"` request returns a `tool_calls` array naming a `finish`
  function. Degrades gracefully (a FAIL row, no abort) against a server that
  lacks the flags.

### Changed

### Fixed

## [0.6.0] - 2026-05-27

### Added

- **devague workflow trio** vendored under `.claude/skills/` (cite-don't-import):
  `think` (idea→spec), `spec-to-plan` (spec→plan), and `assign-to-workforce`
  (plan→parallel implementation) — the operator chain for the deterministic
  `devague` CLI. Authored in `agentculture/devague`, vendored via guildmaster;
  each carries `type: command` (load-bearing on the culture/agex backend, where
  a `SKILL.md` without `type:` is silently skipped). They drive the `devague`
  CLI at runtime (`uv tool install devague`), resolved portably by the wrappers.
- **`docs/skill-sources.md`** — provenance ledger recording the citation path
  and authoring origin of every vendored skill (the trio plus the six
  steward-sourced skills).

## [0.5.0] - 2026-05-27

Redesigned the repo around **running, assessing, and switching the local vLLM
model**. The model-ops logic that lived in the `model-runner` *skill* is now a
first-class CLI. lepenseur is still the deployed agent that consumes the served
model; model-gear is the tool that runs it.

### Added

- **Model-ops verbs** on the `model` CLI: `switch <model>`, `serve` (alias
  `start`) / `stop`, `status`, `assess` (correctness probes), `benchmark`
  (decode throughput + prefill), and `init` (scaffold a deployment dir). Write
  verbs (`switch`/`serve`/`stop`/`init`) are **dry-run by default** and require
  `--apply` (mutation-safety rule).
- **Scaffold-based deployment.** `docker-compose.yml` + `env.example` ship as
  packaged templates under `model_gear/templates/`; `model init` materialises
  them into `~/.model-gear` (default), a `TARGET`, or the local folder. Every
  model-ops verb resolves the deployment dir via `--compose-dir` →
  `$MODEL_GEAR_DIR` → `~/.model-gear`.
- Ported runtime modules (`model_gear/runtime/` + `model_gear/assess.py`),
  stdlib-only (`urllib`, fixed-argv `subprocess`), with full unit tests.
- `model overview` now folds in the currently-served model and the
  candidate-model list, filterable with `--current` / `--list`.

### Changed

- **PyPI distribution renamed `lepenseur` → `model-gear`; binary `lepenseur` →
  `model`; Python package `lepenseur` → `model_gear`.** Error class
  `LepenseurError` → `ModelGearError`. The `lepenseur` console script is removed.
- Agent-first verbs reframed for the tool: `whoami` reports tool/machine/served
  model/container health/agent; `learn` teaches the model-ops surface; `explain`
  catalog rewritten (`switch`/`assess`/`backend`/`models`/…).
- `doctor` is now **real** — checks docker availability, deployment scaffold,
  `.env` ↔ `culture.yaml` coherence, and `/health` reachability (a down model is
  a warning, not a failure).
- The `model-runner` skill is now a thin shim that `exec`s `model`; its
  `_assess.py` was removed (the logic lives in `model_gear/assess.py`).
- `AGENTS.md` / `culture.yaml` clarified: they describe the deployed `lepenseur`
  agent, not the repo. README + CLAUDE.md reoriented around model-gear.

### Fixed

- **BREAKING:** the vLLM container is renamed `lepenseur-vllm` → `model-gear-vllm`.
  A box running the old container must `docker compose down` under the old name,
  then `model init --apply` + `model serve --apply`.

## [0.4.0] - 2026-05-27

### Added

- `model-runner` skill (local, not vendored): `switch` the local vLLM runtime
  model and `assess`/benchmark it (stdlib `_assess.py` for correctness +
  throughput, host-side facts via the wrapper). Drives this repo's compose +
  `.env`; documented in CLAUDE.md and README. Mutating verbs (`switch`, `down`)
  are dry-run by default and require `--apply` (CLAUDE.md mutation-safety rule);
  `--port` defaults to `.env`'s `VLLM_PORT` (then 8000).

### Changed

- `docs/qwen3.6-27b-nvfp4.md`: filled with the live load-test (DGX Spark/GB10,
  2026-05-27). `mmangkad/Qwen3.6-27B-NVFP4` loads and serves under our vLLM image
  (no `--trust-remote-code`); ~7.9–8.0 tok/s decode, ~70 GB reserved, 29 GB
  weights. It is a hybrid Mamba/linear-attention vision-language model and is
  slower on decode than the 32B here — recommendation: **keep the 32B**. All
  pre-flight caveats (SGLang-only, multimodal, ModelOpt rc) validated/resolved.

## [0.3.0] - 2026-05-27

### Added

- `docs/qwen3-32b-nvfp4.md`: per-model doc for the current runtime model, with a
  live test on DGX Spark (GB10) — `nvcr.io/nvidia/vllm:26.04-py3` (engine
  `0.19.0+...nv26.04`), ~9.7 tok/s decode (batch=1), ~2,800 tok/s prefill, ~72 GB
  reserved at `gpu-memory-utilization=0.6`, correctness verified.
- `docs/qwen3.6-27b-nvfp4.md`: per-model doc for candidate
  `mmangkad/Qwen3.6-27B-NVFP4`. Its `Qwen3_5ForConditionalGeneration` arch is
  registered in the current vLLM image (so the same compose can serve it); live
  load-test/benchmark tracked by issue #6.
- README "Per-model notes" linking both docs.

### Fixed

- `docker-compose.yml`: corrected the `--reasoning-parser=qwen3` comment — on the
  nv26.04 build the `<think>` trace is returned in the `reasoning` field, not
  `reasoning_content`.

## [0.2.0] - 2026-05-27

### Added

- `docker-compose.yml` + `.env.example`: a local vLLM server (NGC
  `nvcr.io/nvidia/vllm` image) that serves the runtime model as an
  OpenAI-compatible API on `:8000` for the `acp` backend, tuned for DGX Spark
  (GB10 Blackwell, 128 GB unified memory).
- README "Running the model locally (vLLM)" section.

### Changed

- Switched lepenseur's runtime model from
  `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` to `nvidia/Qwen3-32B-NVFP4`
  across `culture.yaml`, `AGENTS.md`, `lepenseur/explain/catalog.py`, `README.md`,
  and `CLAUDE.md` (32B dense NVFP4 reasoning model with a thinking mode).

## [0.1.0] - 2026-05-22

### Added

- Initial CLI/PyPI sibling scaffold (copied and adapted from the `lecodeur`
  twin): top-level `lepenseur` package with the `lepenseur` console script.
- Read-only verbs: `whoami`, `learn`, `explain`, `overview`, and a `cli`
  noun with `cli overview`.
- `doctor` verb shipped as a rubric-shaped stub; real self-diagnosis semantics
  for a thinking ("non-doer") agent are deferred to a follow-up.
- Runtime identity files: `AGENTS.md` and `culture.yaml` (acp backend,
  `vllm-local/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`).
- CI: `tests.yml` (test + lint + `afi cli doctor . --strict` gate +
  version-check) and `publish.yml` (PyPI/TestPyPI via Trusted Publishing).
- Six vendored skills under `.claude/skills/` (cicd, communicate, version-bump,
  run-tests, sonarclaude, doc-test-alignment), provenance: steward.
