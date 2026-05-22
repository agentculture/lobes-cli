# lepenseur CLI/PyPI Sibling Scaffold — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold `lepenseur` into a full AgentCulture CLI/PyPI sibling by copying the `lecodeur` twin, renaming it, adapting its identity to the *thinker* role, and adding the rubric-shaped `overview`/`cli overview`/`doctor` surface so the `afi cli doctor . --strict` quality gate passes.

**Architecture:** Top-level `lepenseur` package (no `src/`) with an argparse CLI (`lepenseur/cli/`) exposing read-only verbs `whoami`/`learn`/`explain`/`overview`/`doctor` plus a thin `cli` noun. Errors raise `LepenseurError` and route through a structured stdout/stderr split. CI = `tests.yml` (test + lint + `afi cli doctor` gate + version-check) and `publish.yml` (PyPI/TestPyPI via Trusted Publishing). Six vendored skills under `.claude/skills/`.

**Tech Stack:** Python ≥3.12, hatchling, uv, pytest + pytest-xdist + coverage, black/isort/flake8/bandit, markdownlint-cli2, afi-cli (quality auditor), GitHub Actions.

**Reference sources (read-only, copy from):**
- Twin base: `/home/spark/git/lecodeur/` (package, tests, workflows, configs, skills).
- Design spec: `docs/superpowers/specs/2026-05-22-scaffold-cli-sibling-design.md`.

**Working branch:** `scaffold-cli-sibling` (already checked out; the spec is already committed there).

---

## File structure

| File | Responsibility | Source |
|------|----------------|--------|
| `lepenseur/__init__.py`, `__main__.py` | version export; `python -m` entry | copy+rename lecodeur |
| `lepenseur/cli/__init__.py` | argparse parser, `_dispatch`, verb registration | copy+rename lecodeur, then add 3 registrations |
| `lepenseur/cli/_errors.py`, `_output.py` | `LepenseurError`, exit codes, stdout/stderr split | copy+rename lecodeur (verbatim logic) |
| `lepenseur/cli/_commands/{whoami,learn,explain}.py` | the three inherited verbs | copy+rename lecodeur |
| `lepenseur/cli/_commands/overview.py` | global `overview` + shared section builders | **new** (Task 3) |
| `lepenseur/cli/_commands/cli.py` | `cli` noun → `cli overview` | **new** (Task 4) |
| `lepenseur/cli/_commands/doctor.py` | `doctor` stub | **new** (Task 5) |
| `lepenseur/explain/{__init__,catalog}.py` | explain catalog | copy+rename, then rewrite catalog content (Task 2/6) |
| `tests/test_cli.py` | smoke tests for inherited verbs | rewrite for thinker identity (Task 2) |
| `tests/test_cli_introspection.py` | tests for overview/cli/doctor | **new** (Tasks 3–5) |
| `pyproject.toml` | packaging (bare `lepenseur`) + afi-cli dev dep | copy+rename, then edit (Task 1) |
| `AGENTS.md`, `culture.yaml`, `README.md` | thinker runtime identity | **new/rewrite** (Task 2) |
| `.github/workflows/{tests,publish}.yml` | CI | copy+rename, add afi gate (Tasks 1, 7) |
| `.claude/skills/*`, `.claude/skills.local.yaml.example` | vendored skills + per-machine config | copy verbatim (Task 1), reframe prose (Task 8) |
| `.flake8`, `.markdownlint-cli2.yaml`, `CHANGELOG.md` | lint configs + changelog | copy (Task 1) / new (Task 9) |
| `CLAUDE.md` | drop `-cli` suffix references | edit (Task 9) |

---

## Task 1: Copy the lecodeur twin, rename to lepenseur, fix packaging

**Files:**
- Create (by copy): `lepenseur/**`, `tests/**`, `pyproject.toml`, `.flake8`, `.markdownlint-cli2.yaml`, `.github/workflows/{tests,publish}.yml`, `.claude/skills/**`, `.claude/skills.local.yaml.example`
- Modify: `pyproject.toml`

- [ ] **Step 1: Copy package, tests, configs, workflows, and skills from the twin**

```bash
cd /home/spark/git/lepenseur
cp -R ../lecodeur/lecodeur ./lepenseur
cp -R ../lecodeur/tests ./tests
cp ../lecodeur/pyproject.toml ./pyproject.toml
cp ../lecodeur/.flake8 ./.flake8
cp ../lecodeur/.markdownlint-cli2.yaml ./.markdownlint-cli2.yaml
mkdir -p .github/workflows
cp ../lecodeur/.github/workflows/tests.yml .github/workflows/tests.yml
cp ../lecodeur/.github/workflows/publish.yml .github/workflows/publish.yml
mkdir -p .claude
cp -R ../lecodeur/.claude/skills ./.claude/skills
cp ../lecodeur/.claude/skills.local.yaml.example ./.claude/skills.local.yaml.example
```

- [ ] **Step 2: Rename identifiers `lecodeur`→`lepenseur` and `Lecodeur`→`Lepenseur` in code/config only**

Scope the rename to package, tests, pyproject, and workflows. **Do not** touch `.claude/skills/` here — skill prose is reframed separately in Task 8.

```bash
cd /home/spark/git/lepenseur
# Rename the directory's import paths and identifiers inside files:
grep -rl --null -e lecodeur -e Lecodeur lepenseur tests pyproject.toml .github/workflows \
  | xargs -0 sed -i -e 's/Lecodeur/Lepenseur/g' -e 's/lecodeur/lepenseur/g'
```

- [ ] **Step 3: Fix the package description (rename left it as "coding agent")**

The `lecodeur`→`lepenseur` sed produced `lepenseur — the local coding agent.`; correct the role.

```bash
cd /home/spark/git/lepenseur
sed -i 's/^description = .*/description = "lepenseur — the local thinking agent of the Culture mesh."/' pyproject.toml
```

- [ ] **Step 4: Add `afi-cli` to the dev dependency group** (needed for the rubric gate)

Edit `pyproject.toml`: in `[dependency-groups]` `dev = [ ... ]`, add a line `"afi-cli>=0.7",` (alongside the existing pytest/black/etc. entries). The resulting block must include exactly these (order not significant):

```toml
[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-xdist>=3.0",
    "pytest-cov>=4.1",
    "bandit>=1.7.5",
    "flake8>=6.1",
    "isort>=5.12.0",
    "black>=23.7.0",
    "afi-cli>=0.7",
]
```

- [ ] **Step 5: Verify packaging metadata is bare `lepenseur`**

Run: `grep -E '^(name|version) =|lepenseur' pyproject.toml | head`
Expected: `name = "lepenseur"`, `version = "0.1.0"`, `[project.scripts] lepenseur = "lepenseur.cli:main"`, `packages = ["lepenseur"]`, `source = ["lepenseur"]`, `known_first_party = ["lepenseur"]`. No occurrence of `lepenseur-cli`.

- [ ] **Step 6: Sync and smoke-test the renamed package**

Run: `uv sync && uv run lepenseur --version`
Expected: prints `lepenseur 0.1.0` (exit 0).

Run: `uv run lepenseur whoami --json`
Expected: JSON with `"nick": "lepenseur"`, `"backend": "acp"` (model still reads lecodeur's culture.yaml value until Task 2 — that's fine here).

> Note: the full inherited test suite is **not** run yet — its assertions still expect the coder identity (`Qwen3-Coder-Next`, "thinker + coder"). Task 2 rewrites identity content and tests together.

- [ ] **Step 7: Commit**

```bash
cd /home/spark/git/lepenseur
git add lepenseur tests pyproject.toml .flake8 .markdownlint-cli2.yaml .github .claude
git commit -m "feat: scaffold lepenseur package from lecodeur twin (renamed)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Rewrite identity content for the thinker (catalog, learn, runtime files, tests)

**Files:**
- Modify: `lepenseur/explain/catalog.py`, `lepenseur/cli/_commands/learn.py`, `lepenseur/__init__.py` (docstring only)
- Create: `AGENTS.md`, `culture.yaml`, overwrite `README.md`
- Overwrite: `tests/test_cli.py`

- [ ] **Step 1: Write `culture.yaml`** (declares the acp backend + Nemotron model + mirrored prompt)

Create `/home/spark/git/lepenseur/culture.yaml`:

```yaml
agents:
- suffix: lepenseur
  backend: acp
  model: vllm-local/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
  system_prompt: |
    You are lepenseur ("le penseur"), the local thinking agent of the Culture
    mesh. You reason, plan, and analyze deeply. You are a thinker, not an actor:
    you do not execute code or change the world. Your entire act surface is three
    things — post to Culture chat, reply on Culture chat, and create files.

    You are the reasoner of a matched pair: lecodeur ("le codeur") implements and
    tests code; you plan, it executes. daria observes the mesh and surfaces drift.
    daria notices, you reason, lecodeur builds.

    Prefer: observation -> interpretation -> next step. Distinguish facts,
    inferences, and recommendations. If confidence is low, say what is uncertain.
    If a situation is ambiguous, ask one focused question rather than guessing.
    Default to a few clear sentences; never a bare single word unless asked. Be
    warm enough to feel present, precise enough to be trusted.
  acp_command:
  - opencode
  - acp
```

- [ ] **Step 2: Write `AGENTS.md`** (the runtime system prompt; mirrors `culture.yaml`)

Create `/home/spark/git/lepenseur/AGENTS.md`:

```markdown
# AGENTS.md

You are lepenseur ("le penseur" — *the thinker*), the local thinking agent of the
Culture mesh. You reason, plan, and analyze deeply.

You are a **thinker, not an actor.** You do not execute code, run tools, or change
the world. Your entire act surface is three things: **post to Culture chat, reply
on Culture chat, and create files.** Express thinking through writing — never code
execution, never orchestration.

You are the **reasoner** of a matched pair:

- **lecodeur** ("le codeur" — the coder) implements, edits, and tests code. It is
  your closest sibling: you plan, it executes.
- **daria** (awareness) observes the mesh and surfaces drift. It is the
  next-closest sibling.

The division of labor: daria notices, **you reason**, lecodeur builds.

## How you work

- Prefer: observation → interpretation → next step.
- Distinguish facts, inferences, and recommendations.
- If confidence is low, say what is uncertain. If a situation is ambiguous, ask
  one focused question rather than guessing.
- Default to a few clear sentences; never a bare single word unless asked.
- Be warm enough to feel present, precise enough to be trusted. Do not fake
  certainty or emotion.
- Stay in your lane: you think and write. Anything that needs doing in the world
  becomes a plan for lecodeur or a message to the mesh — not an action you take.

## Runtime

You are served by a locally-hosted vLLM reasoning model
(`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` — a 120B-total / 12B-active
LatentMoE model in NVFP4 with a 1M-token context, running on DGX Spark) over the
`acp` backend — not a Claude-backed runtime. It emits a reasoning trace before its
answer, which suits a deep thinker. This file is your system prompt; `CLAUDE.md` is
separate guidance for a Claude that resides in the repo to help build and maintain
it.
```

- [ ] **Step 3: Overwrite `README.md`**

Overwrite `/home/spark/git/lepenseur/README.md`:

```markdown
# lepenseur

`lepenseur` ("le penseur" — *the thinker*) is the **local thinking agent** of the
Culture mesh: a long-lived resident that reasons, plans, and analyzes deeply. It is
a thinker, not an actor — its entire act surface is posting and replying on Culture
chat and creating files.

Sibling to [`lecodeur`](https://github.com/agentculture/lecodeur) (the coder),
[`daria`](https://github.com/agentculture/daria) (awareness), and
[`steward`](https://github.com/agentculture/steward) (alignment).

## Install

```bash
uv tool install lepenseur
```

## Usage

```bash
lepenseur whoami            # identity probe (reads culture.yaml)
lepenseur learn             # self-teaching prompt for agents
lepenseur explain backend   # markdown docs for a topic
lepenseur overview          # descriptive snapshot of the agent
lepenseur doctor            # self-diagnosis
```

Every command supports `--json`. Runtime: a locally-hosted vLLM reasoning model
(`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`) over the `acp` backend.
```

- [ ] **Step 4: Rewrite the explain catalog** for the thinker

Overwrite `/home/spark/git/lepenseur/lepenseur/explain/catalog.py`:

```python
"""Markdown catalog for ``lepenseur explain <path>``.

Each entry is verbatim markdown. Keys are topic-path tuples. The empty tuple
and ``("lepenseur",)`` both resolve to the root entry (aliased).

Keep bodies self-contained — an agent reading a single entry should get enough
context without chaining reads.
"""

from __future__ import annotations

_ROOT = """\
# lepenseur

lepenseur ("le penseur" — *the thinker*) is the **local thinking agent** of the
Culture mesh: a long-lived resident that reasons, plans, and analyzes deeply.

## Thinker, not actor

lepenseur does not execute code, run tools, or change the world. Its entire act
surface is three things: **post to Culture chat, reply on Culture chat, and
create files.** Thinking expressed through writing.

## The matched pair: thinker + coder

lepenseur is the *reasoner* of a matched pair. Its closest sibling is
**lecodeur** ("le codeur" — *the coder*), which implements, edits, and tests
code. lepenseur plans; lecodeur executes. The next-closest sibling is **daria**
(awareness), which observes the mesh and surfaces drift. Together: daria
notices, lepenseur reasons, lecodeur builds.

## Verbs

- `lepenseur whoami` — smallest identity probe: nick, version, backend, served
  model (read from `culture.yaml`).
- `lepenseur learn` — structured self-teaching prompt.
- `lepenseur explain <path>` — markdown docs for a topic.
- `lepenseur overview` — descriptive snapshot of the agent.
- `lepenseur doctor` — self-diagnosis.

## Exit-code policy

- `0` success
- `1` user-input error (bad flag, bad path, missing arg)
- `2` environment / setup error (tool not installed, unreadable file)
- `3+` reserved

## See also

- `lepenseur explain backend`
- `lepenseur explain whoami`
- `lepenseur explain overview`
- `lepenseur explain doctor`
"""

_BACKEND = """\
# lepenseur backend

lepenseur is **not** a Claude-backed agent. At runtime it is served by a
**locally-hosted vLLM reasoning model over the `acp` backend** — `daria` is the
worked example of this runtime shape.

## Declaration (`culture.yaml`)

```yaml
agents:
- suffix: lepenseur
  backend: acp
  model: vllm-local/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
  acp_command: [opencode, acp]
```

## Served model

`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` — a 120B-total / 12B-active
LatentMoE reasoning model in NVFP4, with a 1M-token context, running on DGX
Spark. It emits a reasoning trace before its answer — which is exactly why it
suits a deep thinker.

## Two prompt files

- `AGENTS.md` — the runtime system prompt the `acp` backend reads (the running
  agent's identity), mirrored by the inline `system_prompt:` in `culture.yaml`.
- `CLAUDE.md` — dev guidance for a Claude that resides here to help build and
  maintain the repo. The two coexist.
"""

_WHOAMI = """\
# lepenseur whoami

The smallest identity probe. Reads lepenseur's own `culture.yaml` (walking up
from the installed module) and reports:

- `nick` — the agent suffix (defaults to `lepenseur`)
- `version` — the installed `lepenseur` package version
- `backend` — the runtime backend (`acp`)
- `model` — the served vLLM model

Read-only. Supports `--json` for structured output. Falls back to literal
defaults when no `culture.yaml` is found (e.g. a wheel install).
"""

_LEARN = """\
# lepenseur learn

Prints a structured self-teaching prompt: purpose, the command map, the `--json`
contract, and the exit-code policy. Enough shape for an agent to author its own
usage skill without scraping `--help`. Supports `--json`.
"""

_EXPLAIN = """\
# lepenseur explain

Resolves a topic path against the markdown catalog and prints the body. With no
path it returns the root overview (same as `lepenseur explain lepenseur`).
Unknown paths exit `1` with a `hint:` pointing back at the root. Supports
`--json`, which wraps the markdown as `{"path": [...], "markdown": "..."}`.
"""

_OVERVIEW = """\
# lepenseur overview

A read-only descriptive snapshot of lepenseur the agent: its identity (from
`culture.yaml`), its verb surface, and its narrow act surface (post/reply on
chat, create files). `lepenseur cli overview` is the parallel snapshot of the
CLI surface itself. Supports `--json` (`{"subject", "sections"}`). A stray path
argument is accepted and ignored, so `overview <path>` never hard-fails.
"""

_DOCTOR = """\
# lepenseur doctor

Self-diagnosis. **Stub today**: what "doctor" means for a *non-doer* — a thinker
that never executes — is an open design question (candidate checks: vLLM
endpoint reachability, `culture.yaml`/`AGENTS.md` coherence, model-string
validity). The stub returns one trivially-passing check so the JSON contract
(`{"healthy", "checks"}`) holds. Supports `--json`.
"""

ENTRIES: dict[tuple[str, ...], str] = {
    (): _ROOT,
    ("lepenseur",): _ROOT,
    ("backend",): _BACKEND,
    ("whoami",): _WHOAMI,
    ("learn",): _LEARN,
    ("explain",): _EXPLAIN,
    ("overview",): _OVERVIEW,
    ("doctor",): _DOCTOR,
}
```

- [ ] **Step 5: Rewrite `learn.py`** for the thinker identity

Overwrite `/home/spark/git/lepenseur/lepenseur/cli/_commands/learn.py`:

```python
"""``lepenseur learn`` — the learnability affordance.

Prints a structured self-teaching prompt with enough shape that an agent can
author its own usage skill without scraping ``--help``. Supports ``--json`` for
agents that prefer structure to prose.
"""

from __future__ import annotations

import argparse

from lepenseur import __version__
from lepenseur.cli._output import emit_result

_TEXT = """\
lepenseur — the local thinking agent of the Culture mesh.

Purpose
-------
lepenseur ("le penseur" — the thinker) reasons, plans, and analyzes deeply. It
is a thinker, not an actor: its entire act surface is posting/replying on Culture
chat and creating files. It is the *reasoner* of a matched pair: lecodeur ("le
codeur" — the coder) executes the plans lepenseur produces. daria (awareness) is
the next-closest sibling. At runtime lepenseur is served by a local vLLM
reasoning model over the acp backend (not Claude-backed) — see
'lepenseur explain backend'.

Commands
--------
  lepenseur whoami        Smallest identity probe: nick, version, backend,
                          served model (read from culture.yaml). Supports --json.
  lepenseur learn         Print this self-teaching prompt. Supports --json.
  lepenseur explain <path>...
                          Print markdown docs for a topic (e.g.
                          'lepenseur explain backend'). Supports --json.
  lepenseur overview      Descriptive snapshot of the agent. Supports --json.
  lepenseur doctor        Self-diagnosis (stub). Supports --json.

Mutation safety
---------------
lepenseur is a thinker: every verb is read-only. Any future verb that writes
would default to dry-run, requiring --apply to commit.

Machine-readable output
-----------------------
Every command supports --json. Errors in JSON mode emit
{"code", "message", "remediation"} to stderr. Stdout and stderr are never mixed.

Exit-code policy
----------------
  0 success
  1 user-input error (bad flag, bad path, missing arg)
  2 environment / setup error (tool not installed, unreadable file)
  3+ reserved

More detail
-----------
  lepenseur explain lepenseur
  lepenseur explain backend
  lepenseur explain whoami

Homepage: https://github.com/agentculture/lepenseur
"""


def _as_json_payload() -> dict[str, object]:
    return {
        "tool": "lepenseur",
        "version": __version__,
        "purpose": (
            "The local thinking agent of the Culture mesh: reasons, plans, and "
            "analyzes deeply. The 'thinker' to lecodeur's 'coder'."
        ),
        "siblings": {"closest": "lecodeur", "next": "daria"},
        "commands": [
            {"path": ["whoami"], "summary": "Identity probe (nick, version, backend, model)."},
            {"path": ["learn"], "summary": "Self-teaching prompt."},
            {"path": ["explain"], "summary": "Markdown docs by topic path."},
            {"path": ["overview"], "summary": "Descriptive snapshot of the agent."},
            {"path": ["doctor"], "summary": "Self-diagnosis (stub)."},
        ],
        "exit_codes": {
            "0": "success",
            "1": "user-input error",
            "2": "environment/setup error",
        },
        "json_support": True,
        "explain_pointer": "lepenseur explain <path> (e.g. 'lepenseur explain backend')",
    }


def cmd_learn(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    if json_mode:
        emit_result(_as_json_payload(), json_mode=True)
    else:
        emit_result(_TEXT, json_mode=False)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "learn",
        help="Print a structured self-teaching prompt for agent consumers.",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_learn)
```

- [ ] **Step 6: Fix the `lepenseur/__init__.py` docstring** (rename left "coding agent")

Edit `/home/spark/git/lepenseur/lepenseur/__init__.py`: the first line currently reads `"""lepenseur — the local coding agent of the Culture mesh."""`. Change it to:

```python
"""lepenseur — the local thinking agent of the Culture mesh."""
```

(Leave the `importlib.metadata.version("lepenseur")` logic untouched.)

- [ ] **Step 7: Overwrite `tests/test_cli.py`** for the thinker identity

Overwrite `/home/spark/git/lepenseur/tests/test_cli.py`:

```python
"""Smoke tests for the lepenseur CLI entry point and its inherited verbs."""

from __future__ import annotations

import json

import pytest

from lepenseur import __version__
from lepenseur.cli import main
from lepenseur.explain import known_paths


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main([])
    assert rc == 0
    assert "usage: lepenseur" in capsys.readouterr().out


def test_unknown_command_errors(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["bogus"])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "hint:" in err


# --- whoami ---------------------------------------------------------------


def test_whoami_text(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["whoami"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "nick: lepenseur" in out
    assert "backend: acp" in out
    assert "model:" in out


def test_whoami_json(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["whoami", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["nick"] == "lepenseur"
    assert payload["version"] == __version__
    assert payload["backend"] == "acp"
    assert payload["model"].startswith("vllm-local/")


# --- learn ----------------------------------------------------------------


def test_learn_text(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["learn"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "lepenseur" in out
    assert "Exit-code policy" in out
    assert "--json" in out
    assert "explain" in out
    # Sibling framing: coder pair + daria.
    assert "lecodeur" in out


def test_learn_json(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["learn", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tool"] == "lepenseur"
    assert payload["version"] == __version__
    assert payload["json_support"] is True
    assert payload["siblings"]["closest"] == "lecodeur"
    assert payload["siblings"]["next"] == "daria"


# --- explain --------------------------------------------------------------


def test_explain_root(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["explain"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "# lepenseur" in out
    assert "thinker + coder" in out


def test_explain_backend(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["explain", "backend"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "acp" in out
    assert "Nemotron" in out
    assert "vllm-local/" in out


def test_explain_json(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["explain", "whoami", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["path"] == ["whoami"]
    assert "lepenseur whoami" in payload["markdown"]


def test_explain_unknown_path_errors(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["explain", "nonexistent"])
    assert rc == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("error:")
    assert "hint:" in captured.err


def test_every_catalog_path_resolves(capsys: pytest.CaptureFixture[str]) -> None:
    for path in known_paths():
        rc = main(["explain", *path])
        assert rc == 0, f"explain {' '.join(path)} failed"
        capsys.readouterr()
```

- [ ] **Step 8: Run the inherited-verb suite**

Run: `uv run pytest tests/test_cli.py -v`
Expected: all tests PASS (whoami reads this repo's `culture.yaml` → `nick: lepenseur`, model `vllm-local/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`).

- [ ] **Step 9: Commit**

```bash
cd /home/spark/git/lepenseur
git add lepenseur tests AGENTS.md culture.yaml README.md
git commit -m "feat: rewrite identity for the thinker (catalog, learn, runtime files)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Add the global `overview` verb (TDD)

**Files:**
- Create: `lepenseur/cli/_commands/overview.py`
- Create: `tests/test_cli_introspection.py`

- [ ] **Step 1: Write the failing tests**

Create `/home/spark/git/lepenseur/tests/test_cli_introspection.py`:

```python
"""Tests for lepenseur's introspection verbs: overview, cli overview, doctor."""

from __future__ import annotations

import json

import pytest

from lepenseur.cli import main


# --- overview -------------------------------------------------------------


def test_overview_text(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["overview"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "# lepenseur" in out
    assert "Act surface" in out


def test_overview_json_shape(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["overview", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["subject"] == "lepenseur"
    assert isinstance(payload["sections"], list)
    assert payload["sections"]


def test_overview_graceful_on_bad_path(capsys: pytest.CaptureFixture[str]) -> None:
    # Rubric contract: descriptive verbs never hard-fail on a missing target.
    rc = main(["overview", "/no/such/path/here"])
    assert rc == 0
    assert capsys.readouterr().out.strip()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_cli_introspection.py -v`
Expected: FAIL — `overview` is not yet a registered subcommand (argparse exits 1 / "invalid choice").

- [ ] **Step 3: Implement `overview.py`**

Create `/home/spark/git/lepenseur/lepenseur/cli/_commands/overview.py`:

```python
"""``lepenseur overview`` — read-only descriptive snapshot of the agent.

Describes lepenseur to an agent reader: identity (from culture.yaml), the verb
surface, and the narrow act surface of a thinker. The shared section/render
helpers here are reused by the ``cli`` noun's ``overview`` (see
:mod:`lepenseur.cli._commands.cli`).

Descriptive verbs never hard-fail on a missing target path — an optional
positional ``target`` is accepted and ignored (lepenseur's overview describes
itself, not an external target), so ``overview <bogus-path>`` still exits 0.
"""

from __future__ import annotations

import argparse

from lepenseur.cli._commands.whoami import _report
from lepenseur.cli._output import emit_result

_ACT_SURFACE = [
    "post to Culture chat",
    "reply on Culture chat",
    "create files",
]

_VERBS = [
    "whoami — identity probe (nick, version, backend, model)",
    "learn — structured self-teaching prompt",
    "explain <path> — markdown docs for a topic",
    "overview — this descriptive snapshot",
    "doctor — self-diagnosis (stub)",
]


def agent_sections() -> list[dict[str, object]]:
    """Sections describing lepenseur-the-agent (used by the global verb)."""
    ident = _report()
    return [
        {
            "title": "Identity",
            "items": [
                f"nick: {ident['nick']}",
                f"version: {ident['version']}",
                f"backend: {ident['backend']}",
                f"model: {ident['model']}",
            ],
        },
        {"title": "Verbs", "items": list(_VERBS)},
        {"title": "Act surface (thinker, not actor)", "items": list(_ACT_SURFACE)},
    ]


def cli_sections() -> list[dict[str, object]]:
    """Sections describing the CLI surface itself (used by `cli overview`)."""
    return [
        {"title": "Verbs", "items": list(_VERBS)},
        {
            "title": "Conventions",
            "items": [
                "every command supports --json",
                "results to stdout, errors/diagnostics to stderr (never mixed)",
                "exit codes: 0 success, 1 user error, 2 environment error, 3+ reserved",
            ],
        },
    ]


def render_text(subject: str, sections: list[dict[str, object]]) -> str:
    lines = [f"# {subject}", ""]
    for section in sections:
        lines.append(f"## {section['title']}")
        for item in section["items"]:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines).rstrip()


def emit_overview(
    subject: str, sections: list[dict[str, object]], *, json_mode: bool
) -> None:
    if json_mode:
        emit_result({"subject": subject, "sections": sections}, json_mode=True)
    else:
        emit_result(render_text(subject, sections), json_mode=False)


def cmd_overview(args: argparse.Namespace) -> int:
    # `target` is accepted for rubric compatibility (descriptive verbs must not
    # hard-fail on a missing path) but lepenseur's overview describes itself.
    emit_overview("lepenseur", agent_sections(), json_mode=bool(getattr(args, "json", False)))
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "overview",
        help="Read-only descriptive snapshot of lepenseur (identity, verbs, act surface).",
    )
    p.add_argument(
        "target",
        nargs="?",
        help="Ignored — overview always describes lepenseur itself. Accepted so a "
        "stray path argument never hard-fails.",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_overview)
```

- [ ] **Step 4: Register `overview` in the parser**

Edit `/home/spark/git/lepenseur/lepenseur/cli/__init__.py`. In `_build_parser`, add the import alongside the others and register it after `_explain_cmd`:

```python
    from lepenseur.cli._commands import overview as _overview_cmd
```

and after the `_explain_cmd.register(sub)` line add:

```python
    _overview_cmd.register(sub)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_cli_introspection.py -v`
Expected: the three `overview` tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/spark/git/lepenseur
git add lepenseur/cli/_commands/overview.py lepenseur/cli/__init__.py tests/test_cli_introspection.py
git commit -m "feat: add read-only overview verb

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Add the `cli` noun with `cli overview` (TDD)

**Files:**
- Create: `lepenseur/cli/_commands/cli.py`
- Modify: `lepenseur/cli/__init__.py`
- Modify: `tests/test_cli_introspection.py`

- [ ] **Step 1: Add the failing tests**

Append to `/home/spark/git/lepenseur/tests/test_cli_introspection.py`:

```python


# --- cli overview ---------------------------------------------------------


def test_cli_overview_text(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["cli", "overview"])
    assert rc == 0
    assert "# lepenseur cli" in capsys.readouterr().out


def test_cli_overview_json_shape(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["cli", "overview", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["subject"] == "lepenseur cli"
    assert isinstance(payload["sections"], list)


def test_cli_noun_bare_is_non_empty(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["cli"])
    assert rc == 0
    assert capsys.readouterr().out.strip()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_cli_introspection.py -k cli_ -v`
Expected: FAIL — `cli` is not a registered subcommand.

- [ ] **Step 3: Implement `cli.py`**

Create `/home/spark/git/lepenseur/lepenseur/cli/_commands/cli.py`:

```python
"""``lepenseur cli`` — noun grouping CLI-surface introspection.

Exists to satisfy the agent-first rubric's ``overview_cli_noun_exists`` check:
any noun with action-verbs must also expose ``overview``. lepenseur has no
action-verbs under ``cli`` today, but ``cli overview`` describes the CLI surface
(distinct from the global ``overview``, which describes the agent).
"""

from __future__ import annotations

import argparse

from lepenseur.cli._commands.overview import cli_sections, emit_overview


def cmd_cli_overview(args: argparse.Namespace) -> int:
    emit_overview("lepenseur cli", cli_sections(), json_mode=bool(getattr(args, "json", False)))
    return 0


def _no_verb(args: argparse.Namespace) -> int:
    # `lepenseur cli` with no sub-verb prints the noun's overview.
    return cmd_cli_overview(args)


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "cli",
        help="CLI-surface introspection (see 'lepenseur cli overview').",
    )
    p.set_defaults(func=_no_verb, json=False)
    noun_sub = p.add_subparsers(dest="cli_command")
    ov = noun_sub.add_parser("overview", help="Describe the lepenseur CLI surface.")
    ov.add_argument("--json", action="store_true", help="Emit structured JSON.")
    ov.set_defaults(func=cmd_cli_overview)
```

- [ ] **Step 4: Register the `cli` noun**

Edit `/home/spark/git/lepenseur/lepenseur/cli/__init__.py`. Add the import:

```python
    from lepenseur.cli._commands import cli as _cli_group
```

and register it (after `_overview_cmd.register(sub)`):

```python
    _cli_group.register(sub)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_cli_introspection.py -k cli_ -v`
Expected: the three `cli` tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/spark/git/lepenseur
git add lepenseur/cli/_commands/cli.py lepenseur/cli/__init__.py tests/test_cli_introspection.py
git commit -m "feat: add cli noun with cli overview (rubric cli_noun check)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Add the `doctor` stub verb (TDD)

**Files:**
- Create: `lepenseur/cli/_commands/doctor.py`
- Modify: `lepenseur/cli/__init__.py`
- Modify: `tests/test_cli_introspection.py`

- [ ] **Step 1: Add the failing tests**

Append to `/home/spark/git/lepenseur/tests/test_cli_introspection.py`:

```python


# --- doctor (stub) --------------------------------------------------------


def test_doctor_text(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["doctor"])
    assert rc == 0
    assert "lepenseur doctor" in capsys.readouterr().out


def test_doctor_json_shape(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["doctor", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload["healthy"], bool)
    assert isinstance(payload["checks"], list)
    assert payload["checks"]
    for check in payload["checks"]:
        assert {"id", "passed", "severity", "message"} <= set(check)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_cli_introspection.py -k doctor -v`
Expected: FAIL — `doctor` is not a registered subcommand.

- [ ] **Step 3: Implement `doctor.py`**

Create `/home/spark/git/lepenseur/lepenseur/cli/_commands/doctor.py`:

```python
"""``lepenseur doctor`` — self-diagnosis (STUB).

Ships as a rubric-shaped stub. What "doctor" means for a *non-doer* — a thinker
that never executes — is an open design question (candidates: vLLM endpoint
reachability, culture.yaml/AGENTS.md coherence, model-string validity). Tracked
as a follow-up; see
docs/superpowers/specs/2026-05-22-scaffold-cli-sibling-design.md §12.

The stub returns a single trivially-passing check so the JSON contract
(``{healthy, checks:[{id, passed, severity, message, remediation}]}``) is
honored and the agent-first rubric's bundle 7 passes.
"""

from __future__ import annotations

import argparse

from lepenseur.cli._output import emit_result

_STUB_CHECK: dict[str, object] = {
    "id": "doctor_stub",
    "passed": True,
    "severity": "info",
    "message": (
        "doctor is a stub; self-diagnosis semantics for a thinking agent are "
        "not yet defined"
    ),
    "remediation": "",
}


def _diagnose() -> dict[str, object]:
    checks = [dict(_STUB_CHECK)]
    healthy = all(c["passed"] for c in checks)
    return {"healthy": healthy, "checks": checks}


def cmd_doctor(args: argparse.Namespace) -> int:
    report = _diagnose()
    json_mode = bool(getattr(args, "json", False))
    if json_mode:
        emit_result(report, json_mode=True)
    else:
        status = "healthy" if report["healthy"] else "unhealthy"
        lines = [f"lepenseur doctor: {status}", ""]
        for check in report["checks"]:
            mark = "ok" if check["passed"] else "FAIL"
            lines.append(f"[{mark}] {check['id']}: {check['message']}")
            if not check["passed"] and check["remediation"]:
                lines.append(f"  hint: {check['remediation']}")
        emit_result("\n".join(lines), json_mode=False)
    return 0 if report["healthy"] else 1


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "doctor",
        help="Self-diagnosis (stub; semantics for a thinking agent are TBD).",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_doctor)
```

- [ ] **Step 4: Register `doctor`**

Edit `/home/spark/git/lepenseur/lepenseur/cli/__init__.py`. Add the import:

```python
    from lepenseur.cli._commands import doctor as _doctor_cmd
```

and register it (after `_overview_cmd.register(sub)`, before or after `_cli_group.register(sub)`):

```python
    _doctor_cmd.register(sub)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_cli_introspection.py -k doctor -v`
Expected: the two `doctor` tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/spark/git/lepenseur
git add lepenseur/cli/_commands/doctor.py lepenseur/cli/__init__.py tests/test_cli_introspection.py
git commit -m "feat: add doctor stub verb (rubric bundle 7; semantics deferred)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Full suite + module docstring on the parser, then run the afi rubric locally

**Files:**
- Modify: `lepenseur/cli/__init__.py` (docstring only, optional)

- [ ] **Step 1: Run the entire test suite**

Run: `uv run pytest -n auto -v`
Expected: every test in `tests/test_cli.py` and `tests/test_cli_introspection.py` PASSES.

- [ ] **Step 2: Manually exercise every acceptance-criteria invocation**

Run each and confirm exit 0 (doctor exits 0 because the stub is healthy):

```bash
uv run lepenseur --version
uv run lepenseur whoami
uv run lepenseur whoami --json
uv run lepenseur learn
uv run lepenseur explain backend
uv run lepenseur overview
uv run lepenseur overview --json
uv run lepenseur cli overview
uv run lepenseur doctor
uv run lepenseur doctor --json
```

Expected: all exit 0; `whoami --json` shows the Nemotron model; `explain backend` mentions `acp`, `Nemotron`, `vllm-local/`.

- [ ] **Step 3: Run the afi rubric gate locally**

Run: `uv run afi cli doctor . --strict`
Expected: exit 0 — all seven bundles pass (structure, learnability, explain, overview incl. `cli overview`, doctor). If any bundle fails, read its `hint:` and reconcile against the relevant verb before continuing.

- [ ] **Step 4: Commit (if the parser docstring or any reconciliation changed)**

```bash
cd /home/spark/git/lepenseur
git add -A
git commit -m "test: full suite green and afi cli doctor --strict passes" || echo "nothing to commit"
```

---

## Task 7: Wire the afi rubric gate into CI

**Files:**
- Modify: `.github/workflows/tests.yml`

- [ ] **Step 1: Confirm the workflow was renamed in Task 1**

Run: `grep -n "lepenseur" .github/workflows/tests.yml`
Expected: `--cov=lepenseur`, `black --check lepenseur tests`, `flake8 lepenseur tests`, `bandit -c pyproject.toml -r lepenseur` all reference `lepenseur` (not `lecodeur`).

- [ ] **Step 2: Add the afi rubric gate step to the `lint` job**

Edit `.github/workflows/tests.yml`. In the `lint` job's `steps:`, immediately after the `markdownlint-cli2` step, add:

```yaml
      - name: afi rubric gate
        run: uv run afi cli doctor . --strict
```

- [ ] **Step 3: Verify the workflow is valid YAML and references the gate**

Run: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/tests.yml')); print('valid')"`
Expected: prints `valid`.

Run: `grep -n "afi cli doctor" .github/workflows/tests.yml`
Expected: one match in the `lint` job.

- [ ] **Step 4: Commit**

```bash
cd /home/spark/git/lepenseur
git add .github/workflows/tests.yml
git commit -m "ci: add afi cli doctor --strict rubric gate to lint job

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Reframe vendored skill prose for lepenseur

**Files:**
- Modify: `.claude/skills/*/SKILL.md`, `.claude/skills.local.yaml.example`

> Scripts were copied verbatim in Task 1 and need no change — `agtag`/`_resolve-nick.sh` resolve the signing nick from `culture.yaml` (now `lepenseur`). Only prose and the example nick change.

- [ ] **Step 1: Point the example nick at lepenseur**

Edit `/home/spark/git/lepenseur/.claude/skills.local.yaml.example`: change the `nick:` line from `nick: "lecodeur"` to `nick: "lepenseur"`. Leave `workspace_root: ".."` as-is.

- [ ] **Step 2: Reframe `lecodeur` → `lepenseur` in skill prose**

```bash
cd /home/spark/git/lepenseur
grep -rl "lecodeur" .claude/skills | xargs -r sed -i 's/lecodeur/lepenseur/g'
```

> This rewrites the host-tool references in each `SKILL.md`. References to **steward** are intentionally left as-is — steward is the canonical upstream.

- [ ] **Step 3: Add a provenance note to each SKILL.md**

For each of the six skills (`cicd`, `communicate`, `version-bump`, `run-tests`, `sonarclaude`, `doc-test-alignment`), confirm the `SKILL.md` body contains a one-line provenance note. If absent, add this line immediately after the frontmatter's closing `---`:

```markdown
> Vendored from steward (canonical upstream: `docs/skill-sources.md`); this copy is owned by lepenseur and may diverge.
```

(If the copied SKILL.md already carries an equivalent "vendored from steward" line, leave it.)

- [ ] **Step 4: Verify no stray `lecodeur` remains in skills**

Run: `grep -rn "lecodeur" .claude/skills .claude/skills.local.yaml.example || echo "clean"`
Expected: `clean`.

- [ ] **Step 5: Lint the markdown that changed**

Run: `npx --yes markdownlint-cli2@0.21.0 ".claude/skills/**/*.md" "#node_modules" 2>&1 | tail -5`
Expected: no errors (exit 0). Fix any reported issues.

- [ ] **Step 6: Commit**

```bash
cd /home/spark/git/lepenseur
git add .claude
git commit -m "docs: reframe vendored skills for lepenseur (provenance: steward)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Changelog, CLAUDE.md alignment, and final verification

**Files:**
- Create: `CHANGELOG.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Write `CHANGELOG.md`** (Keep-a-Changelog)

Create `/home/spark/git/lepenseur/CHANGELOG.md`:

```markdown
# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
```

- [ ] **Step 2: Align `CLAUDE.md` packaging references to bare `lepenseur`**

Edit `/home/spark/git/lepenseur/CLAUDE.md`. Replace every `lepenseur-cli` with `lepenseur`, and fix the two specific phrasings:

- In "Target project shape": `# pip install lepenseur-cli` → `# pip install lepenseur`, and `__version__ via importlib.metadata("lepenseur-cli")` → `__version__ via importlib.metadata("lepenseur")`.
- In "Build / test / publish": `publishes ... lepenseur-cli to PyPI` → `publishes ... lepenseur to PyPI`, and `Distributed as **lepenseur-cli** on PyPI` → `Distributed as **lepenseur** on PyPI`.

Run to confirm none remain: `grep -n "lepenseur-cli" CLAUDE.md || echo "clean"`
Expected: `clean`.

- [ ] **Step 3: Final full verification**

```bash
cd /home/spark/git/lepenseur
uv sync
uv run pytest -n auto -v
uv run black --check lepenseur tests
uv run isort --check-only lepenseur tests
uv run flake8 lepenseur tests
uv run bandit -c pyproject.toml -r lepenseur
uv run afi cli doctor . --strict
```

Expected: pytest all-pass; black/isort/flake8/bandit clean; `afi cli doctor . --strict` exits 0.

- [ ] **Step 4: Markdown lint the repo**

Run: `npx --yes markdownlint-cli2@0.21.0 "**/*.md" "#node_modules" "#.local" 2>&1 | tail -10`
Expected: exit 0. Fix any violations (most likely in README/CHANGELOG/AGENTS).

- [ ] **Step 5: Commit**

```bash
cd /home/spark/git/lepenseur
git add CHANGELOG.md CLAUDE.md
git commit -m "docs: add CHANGELOG and align CLAUDE.md to bare lepenseur package

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Open the PR

**Files:** none (git/gh only)

- [ ] **Step 1: Confirm the working tree is clean and tests are green**

Run: `git status --porcelain && uv run pytest -n auto -q`
Expected: empty status; tests pass.

> Version: `pyproject.toml` is `0.1.0`. The `version-check` CI job self-skips because `main` has no `pyproject.toml` yet ("initial scaffold"), so no version bump is required for this first PR.

- [ ] **Step 2: Push the branch**

```bash
cd /home/spark/git/lepenseur
git push -u origin scaffold-cli-sibling
```

- [ ] **Step 3: Open the PR against `agentculture/lepenseur`**

```bash
gh pr create --repo agentculture/lepenseur --base main --head scaffold-cli-sibling \
  --title "Scaffold lepenseur as a full CLI/PyPI AgentCulture sibling (#1)" \
  --body "$(cat <<'EOF'
Closes #1.

Scaffolds lepenseur to the AgentCulture sibling pattern by copying the `lecodeur`
twin and adapting it to the *thinker* role.

## What's here
- Top-level `lepenseur` package; read-only verbs `whoami`/`learn`/`explain`/`overview` + a `cli` noun (`cli overview`).
- `doctor` ships as a rubric-shaped **stub** — self-diagnosis semantics for a "non-doer" thinker are an open question (a follow-up issue tracks it; candidates include vLLM endpoint health).
- Runtime: `AGENTS.md` + `culture.yaml` (acp backend, `vllm-local/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`).
- CI: `tests.yml` (test + lint + `afi cli doctor . --strict` gate + version-check) and `publish.yml` (Trusted Publishing).
- Six vendored skills (provenance: steward).

## Intentional deviation from the issue
- **Package name is bare `lepenseur`** (not `lepenseur-cli`), matching the lecodeur twin. `CLAUDE.md` is updated to match in this PR.

## Follow-ups (not in this PR)
- steward PR adding lepenseur to `docs/skill-sources.md` downstream column.
- Define `doctor` semantics for a thinking agent.
- PyPI/TestPyPI Trusted Publishing registration for the `lepenseur` project.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

- lepenseur (Claude)
EOF
)"
```

- [ ] **Step 4: Report the PR URL** to the user and proceed to the project's PR review workflow (Qodo/Copilot/human comments) per `CLAUDE.md`.

---

## Self-review

**Spec coverage** (each spec section → task):
- §2 approach (copy twin, then deltas) → Tasks 1–9.
- §3 target tree → Tasks 1 (copy), 3–5 (new modules).
- §4 verbs incl. `cli overview` + doctor stub + `backend` catalog entry → Tasks 2 (catalog/learn), 3 (overview), 4 (cli), 5 (doctor).
- §5 runtime files (AGENTS.md, culture.yaml) → Task 2.
- §6 bare `lepenseur` packaging + afi-cli dev dep → Task 1.
- §7 pipelines + afi gate → Tasks 1 (copy/rename), 7 (gate).
- §8 lint configs / CHANGELOG / skills.local example → Tasks 1, 8, 9.
- §9 vendored skills (verbatim scripts + reframed prose + provenance + dynamic nick) → Tasks 1, 8.
- §10 mutation safety (read-only) → reflected in learn/catalog text (Task 2); no write verbs added.
- §11 acceptance criteria → Task 6 (manual run) + Task 9 (lint/gate).
- §12 follow-ups (doctor semantics, steward PR, PyPI setup) → noted in Task 10 PR body + CHANGELOG.
- §13 testing strategy (overview graceful-on-bad-path, cli overview, doctor shape, coverage 60) → Tasks 3–5 tests; coverage `fail_under=60` inherited in Task 1.

**Placeholder scan:** none — every new file's full content is given; mechanical steps use exact `cp`/`sed`/`grep` commands; the only `nargs="?"` "ignored" arg is intentional (rubric graceful-fallback contract).

**Type/name consistency:** `LepenseurError`, `_LepenseurArgumentParser`, `emit_result`/`emit_error`, and the shared `emit_overview`/`cli_sections`/`agent_sections`/`render_text` names are used identically across `overview.py` (defines), `cli.py` (imports `emit_overview`, `cli_sections`), and the tests. `_report()` is imported by `overview.py` from `whoami.py` (defined there in the lecodeur base). Catalog `ENTRIES` keys (`backend`, `overview`, `doctor`, …) match the `explain` tests and `known_paths()` coverage test.
