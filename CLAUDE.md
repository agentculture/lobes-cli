# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`lobes` is the tooling that **runs, assesses, and switches** the local,
OpenAI-compatible vLLM model the Culture mesh consumes. The binary is **`lobes`**
(`lobes switch`, `lobes assess`, `lobes serve`, …; `model` is a deprecated alias).

**`lobes` is one identity — the tool *and* the deployed agent:**

- **lobes** is the *repo* and the *tool*. It is a normal CLI/PyPI sibling
  (Python package `lobes`, binary `lobes`, distributed as `lobes-cli`).
- **lobes** is *also* the *agent* deployed *on* the model it serves.
  `AGENTS.md` + `culture.yaml` are that agent's runtime identity (the `acp`
  system prompt and the `suffix: lobes` / `backend: acp` / `model:
  vllm-local/...` declaration). Same name, one identity: the gear runs the model
  and the agent rides on it. (It used to be a separate agent, `lepenseur`; that
  name is retired.)

The served model is **`vllm-local/sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`** (a
Qwen3.6 27B with hybrid Mamba/linear-attention layers, re-exported with its MTP
draft head restored so vLLM speculative decoding (Multi-Token Prediction) works;
text-only (ViT vision tower removed), NVFP4, 256K native context served at the full
256K on the shared DGX Spark; thinking mode with a reasoning trace; ~2.4x
single-stream decode over the archived baseline). lobes runs it; the `acp`
`vllm-local` provider connects the lobes agent to it. (It is the fleet's
default primary. `mmangkad/Qwen3.6-27B-NVFP4` is the archived former primary,
demoted to a candidate but kept — it is the tokenizer source the MTP primary
serves with (`--tokenizer=mmangkad/Qwen3.6-27B-NVFP4`) and the only vision-capable
27B; the `nvidia/Qwen3-32B-NVFP4` dense model also remains a supported candidate —
see `docs/qwen3-32b-nvfp4.md` and `lobes overview --list`.)

Beyond the generate primary, the **fleet** co-resides two tiny pooling gears
behind the gateway, routed by **task family** (`generate` / `embed` / `score` /
`rerank`): an **embedding** gear (`Qwen/Qwen3-Embedding-0.6B` → `POST
/v1/embeddings`, served `--runner pooling --convert embed`) and a **reranker**
gear (`Qwen/Qwen3-Reranker-0.6B` → `POST /v1/rerank` + `/v1/score`, `--convert
classify`). At ~0.6B and `*_GPU_MEM_UTIL=0.06` each they sit alongside the 27B
primary without crowding it; a second *generate* backend (warm fallback) is the
only opt-in piece. See `docs/qwen3-embedding-0.6b.md`,
`docs/qwen3-reranker-0.6b.md`, and `docs/gateway-fleet.md`.

An opt-in **realtime audio overlay** (`lobes init --fleet --audio`) adds an OpenAI
`/v1/audio/*` facade — a `realtime` bridge container (shipped in the wheel as
`lobes.realtime`) that the gateway fans `/v1/audio/*` out to — backed by two
fixed GPU sidecars: **Parakeet** STT (`nvidia/parakeet-tdt-0.6b-v2`, NeMo ASR →
`POST /v1/audio/transcriptions`) and **Chatterbox** TTS (Resemble AI, 0.5B,
Apache-2.0 → `POST /v1/audio/speech`, 24 kHz, zero-shot voice cloning; it replaced
the retired Magpie NIM — no NGC key). These two are hardcoded, **not** in the
switchable catalog (`lobes/catalog.py`). See `docs/realtime-pipeline.md`,
`docs/parakeet-stt.md`, `docs/chatterbox-tts.md`, and `docs/openai-api.md` (the full
OpenAI-compatible endpoint surface). `lobes explain realtime` / `api` are the
in-CLI versions.

## Deployment model

lobes is **scaffold-based, not checkout-based.** The canonical
`docker-compose.yml` + `env.example` are packaged under `lobes/templates/`
and shipped in the wheel. `lobes init` materialises them into a deployment dir —
default **`~/.lobes`**, or a `TARGET` path, or `.` for the local folder.
Every model-ops verb resolves the deployment dir as: `--compose-dir` →
`$LOBES_DIR` → `~/.lobes`, falling back to the legacy `$MODEL_GEAR_DIR` →
`~/.model-gear` when those are set / already scaffolded (so a pre-rename
deployment keeps working). There is no compose file at the repo root.

## CLI surface

```text
lobes/                 # Python package (pip install lobes-cli)
├── __init__.py             # __version__ via importlib.metadata("lobes-cli")
├── __main__.py             # python -m lobes
├── assess.py               # correctness probes + throughput/prefill (stdlib urllib)
├── catalog.py              # the supported-model catalog (the switchable "gears")
├── templates/              # packaged docker-compose*.yml + env.example + Dockerfiles (lobes init)
├── runtime/                # _env (.env r/w) · _compose (dir resolve + docker) · _health · _tunnel (cloudflared)
├── gateway/                # stdlib OpenAI-compatible reverse proxy (the fleet front)
├── realtime/               # /v1/audio/* facade: bridge · tts_client · chatterbox_server · _readiness
├── explain/                # markdown catalog for `lobes explain <path>`
└── cli/
    ├── __init__.py         # argparse main(); registers every verb
    ├── _errors.py          # ModelGearError + EXIT_USER_ERROR / EXIT_ENV_ERROR
    ├── _output.py          # strict stdout/stderr split; --json result emitter
    ├── _runtime_ops.py     # shared glue (deployment dir, port, compose_check)
    └── _commands/          # one module per verb: register(sub) + handler
        ├── switch.py serve.py stop.py status.py assess.py benchmark.py init.py fleet.py
        └── logs.py tunnel.py whoami.py learn.py explain.py overview.py doctor.py cli.py
```

**Mutation safety:** write verbs (`switch`, `serve`, `stop`, `init`, `tunnel`) default to
**dry-run**; require `--apply` to commit. Agents call CLIs in loops, so
safe-by-default is mandatory. The read-only verbs (`status`, `assess`,
`benchmark`, `logs`, `overview`, `whoami`, `explain`, `doctor`) never change the world.

## Build / test / publish

- **Install for dev:** `uv sync`
- **Run CLI from source:** `uv run lobes --version` / `uv run python -m lobes whoami`
- **Tests (all):** `uv run pytest -n auto -v`
- **Single test:** `uv run pytest tests/test_cli_runtime.py::test_name -v`
- **Lint:** `uv run black --check lobes tests`, `uv run isort --check-only lobes tests`, `uv run flake8 lobes tests`, `uv run bandit -c pyproject.toml -r lobes`
- **Rubric gate:** `uv run afi cli doctor . --strict` (CI blocks merge if it fails).
- **Version bump (required every PR):** `python3 .claude/skills/version-bump/scripts/bump.py {patch|minor|major}` — updates `pyproject.toml` and prepends a CHANGELOG entry. The `version-check` CI job **fails the PR if the version equals main's** (AgentCulture every-PR-bumps rule — no exceptions, even for docs/config-only changes). Version is the single source of truth in `pyproject.toml`; `lobes.__version__` is read from package metadata at import.
- **Publish:** push to `main` → `publish.yml` builds with `uv build` and publishes `lobes-cli` to PyPI via Trusted Publishing (no API tokens); `model-gear` is published as a deprecated alias that redirects to `lobes-cli`. PRs publish a `.dev<run_number>` to TestPyPI. Fork PRs are skipped (no OIDC).

## Skills convention

Six skills are vendored from steward (the canonical upstream) under
`.claude/skills/<name>/`: **`cicd`**, **`communicate`**, **`version-bump`**,
**`run-tests`**, **`sonarclaude`**, **`doc-test-alignment`**. This is
*cite-don't-import*: copies are owned by this repo and may diverge from steward.

Three more are vendored from **`agentculture/devague`** (re-broadcast via
guildmaster) — the idea→spec→plan→implementation operator chain for the
deterministic `devague` CLI: **`think`** (idea→spec), **`spec-to-plan`**
(spec→plan), and **`assign-to-workforce`** (plan→parallel implementation). These
three carry **`type: command`** in their frontmatter — load-bearing on the
culture/agex backend (a `SKILL.md` without `type:` is silently skipped when the
repo declares an agent in `culture.yaml`). They depend on the `devague` CLI at
runtime (`uv tool install devague`), resolved portably by the wrappers.

One skill is **local to this repo** (not vendored): **`model-runner`** — a thin
pointer/shim to the `lobes` CLI for switching/serving/assessing the model. The
real implementation is the `lobes` package; the shim `exec`s `lobes`.

The provenance of every vendored skill (citation path + authoring origin) is
recorded in **`docs/skill-sources.md`**.

Each skill ships:

1. `SKILL.md` — *why* and *when* to use it (frontmatter `name` must equal the
   directory name; short prose, no inline 10-step walk-throughs).
2. `scripts/<entry-point>` — the script that automates the workflow.
3. **No external path dependencies.** Scripts must not reach outside this repo.

Per-machine paths live in **`.claude/skills.local.yaml`** (git-ignored); a
committed **`.claude/skills.local.yaml.example`** documents every key. Skills
read the local file and fall back to the example. (The Culture posting nick is
`lobes` — the deployed agent shares the repo/tool name.)

## PR workflow

Every task gets its own branch and PR. Before merging:

1. Wait for all reviewer comments (Qodo, Copilot, humans).
2. Fix valid findings — commit to the same branch.
3. Push back on invalid findings — reply with reasoning.
4. Reply to every thread (fix confirmed or pushback explained).
5. Resolve all threads. **Never merge with unaddressed review comments.**

Bump the version (above) on every PR or CI's `version-check` job fails the run.

## Working with the mesh from here

- **Culture CLI:** `culture` — server lifecycle, agent start/stop, mesh linking.
  Path references assume siblings are checked out alongside this repo
  (`../culture`, `../daria`, `../steward`).
- **steward owns the six steward-sourced skills** (the devague trio is owned
  upstream by `agentculture/devague`; see `docs/skill-sources.md`) and the
  sibling-pattern contract.
  steward files issues on siblings but never edits them — so scaffolding and
  alignment work *for this repo happens in this repo*. steward's
  `docs/skill-sources.md` "Downstream copies" column may still list this repo
  under a retired name (`lepenseur` or `model-gear`); fixing that is a **PR on
  steward**, not an edit from here.
