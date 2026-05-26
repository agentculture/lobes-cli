# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
