# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`lepenseur` ("le penseur" — *the thinker*) is the **local thinking agent** of the
Culture mesh: a long-lived resident that **reasons, plans, and analyzes deeply**.

**It is a thinker, not an actor.** lepenseur does not execute, run tools, or
change the world. Its entire "act" surface is three things: **post to Culture
chat, reply on Culture chat, and create files.** Thinking expressed through
writing — not code execution, not orchestration. Design every affordance around
that constraint: when lepenseur "does" something, it is producing a message or a
document, never an action with side effects beyond those.

It is a sibling to [`culture`](https://github.com/agentculture/culture)
(the IRC-based agent mesh), [`daria`](https://github.com/agentculture/daria) (the
awareness agent), and [`steward`](https://github.com/agentculture/steward) (the
alignment agent) within the Organic Development framework.

**Runtime: local vLLM, NOT a Claude-backed agent.** lepenseur is served by a
locally-hosted vLLM model over the `acp` backend — `daria` is the worked example
of this runtime shape. The model is
**`vllm-local/nvidia/Qwen3-32B-NVFP4`** (a 32B dense reasoning model in NVFP4,
32K-token context extendable to ~131K via YaRN, runs on DGX Spark; it has a
thinking mode and emits a reasoning trace before its answer — which is exactly
why it suits a deep thinker). That backend distinction drives the two prompt files:

- **`AGENTS.md`** — the runtime system prompt the `acp` backend reads. This is
  the *running agent's* identity and behavior. Mirror it with the inline
  `system_prompt:` in `culture.yaml`.
- **`CLAUDE.md`** (this file) — dev guidance for a Claude that resides here to
  *help build and maintain* the repo. The two coexist and serve different
  readers; keep them in sync only where they overlap (the agent's purpose).

## Current state vs. target shape

This repo is an **early scaffold**: the only tracked files today are
`README.md`, `LICENSE`, `.gitignore`, and this `CLAUDE.md`. (A
`.claude/settings.local.json` may exist in a working copy but is git-ignored —
per-machine, not committed.) The structure below is the **target**, defined by
GitHub issue #1 ("Scaffold lepenseur as a
full CLI/PyPI AgentCulture sibling") and by steward's
`docs/sibling-pattern.md` — those two are the authoritative contract for what to
build. When in doubt, read `../steward/docs/sibling-pattern.md` and copy
steward's own tree as the worked example.

## Target project shape (AgentCulture sibling pattern)

Distributed as **`lepenseur`** on PyPI (Trusted Publishing). Python package
is `lepenseur`; the binary is `lepenseur`. Layout follows the afi-cli pattern
(top-level package, no `src/`):

```text
lepenseur/                  # Python package (pip install lepenseur)
├── __init__.py             # __version__ via importlib.metadata("lepenseur")
├── __main__.py             # python -m lepenseur
└── cli/
    ├── __init__.py         # argparse main()
    ├── _errors.py          # LepenseurError + EXIT_USER_ERROR / EXIT_ENV_ERROR
    ├── _output.py          # strict stdout/stderr split; --json result emitter
    └── _commands/          # one module per verb: register(sub) + handler
        ├── whoami.py       # smallest identity probe — reads culture.yaml
        ├── learn.py        # orientation verb (agent affordance)
        └── explain.py      # affordance verb, e.g. `explain backend`
tests/                      # pytest suite (tests/test_cli_*.py)
.claude/skills/<name>/      # SKILL.md + scripts/ per skill (see Skills below)
.github/workflows/          # tests.yml + publish.yml
pyproject.toml              # version source-of-truth (hatchling, Python ≥3.12)
AGENTS.md                   # runtime system prompt (acp backend)
culture.yaml                # backend: acp + model: vllm-local/nvidia/Qwen3-32B-NVFP4
CHANGELOG.md                # Keep-a-Changelog
.flake8, .markdownlint-cli2.yaml   # repo-local lint configs (no home-dir configs)
```

**Mutation safety:** any write verb defaults to **dry-run**; require `--apply`
to commit. Agents call CLIs in loops, so safe-by-default is mandatory. The
initial verbs (`whoami`/`learn`/`explain`) are read-only.

## Build / test / publish

These commands are the convention across siblings; they apply once
`pyproject.toml` and the package exist.

- **Install for dev:** `uv sync`
- **Run CLI from source:** `uv run lepenseur --version` / `uv run python -m lepenseur whoami`
- **Tests (all):** `uv run pytest -n auto -v`
- **Single test:** `uv run pytest tests/test_cli_whoami.py::test_name -v`
- **Lint:** `uv run black --check lepenseur tests`, `uv run isort --check-only lepenseur tests`, `uv run flake8`, `uv run bandit -r lepenseur`
- **Version bump (required every PR):** `python3 .claude/skills/version-bump/scripts/bump.py {patch|minor|major}` — updates `pyproject.toml` and prepends a CHANGELOG entry. The `version-check` CI job **fails the PR if the version equals main's** (AgentCulture every-PR-bumps rule — no exceptions, even for docs/config-only changes). Version is a single source of truth in `pyproject.toml`; `lepenseur.__version__` is read from package metadata at import (no separate literal to sync).
- **Publish:** push to `main` → `publish.yml` builds with `uv build` and publishes `lepenseur` to PyPI via Trusted Publishing (no API tokens). PRs publish a `.dev<run_number>` to TestPyPI. Fork PRs are skipped (no OIDC).

## Skills convention

Six skills are vendored from steward (the canonical upstream) under
`.claude/skills/<name>/`: **`cicd`**, **`communicate`**, **`version-bump`**,
**`run-tests`**, **`sonarclaude`**, **`doc-test-alignment`**. This is
*cite-don't-import*: copies are owned by this repo and may diverge from steward.

Each skill ships:

1. `SKILL.md` — *why* and *when* to use it (frontmatter `name` must equal the
   directory name; short prose, no inline 10-step walk-throughs).
2. `scripts/<entry-point>` — the script that automates the workflow. Following
   a skill should be "run this script," not ten manual steps.
3. **No external path dependencies.** Scripts must not reach into another
   skill's home-directory copy or any path outside this repo — vendor what you
   need. Portability is what lets steward keep siblings aligned.

Per-machine paths live in **`.claude/skills.local.yaml`** (git-ignored); a
committed **`.claude/skills.local.yaml.example`** documents every key. Skills
read the local file and fall back to the example.

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
- **steward owns the six vendored skills** and the sibling-pattern contract.
  steward files issues on siblings but never edits them — so scaffolding and
  alignment work *for this repo happens in this repo*. After vendoring skills,
  the follow-up that adds `lepenseur` to steward's `docs/skill-sources.md`
  "Downstream copies" column must be a **PR on steward**, not an edit from here.
