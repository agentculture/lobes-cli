# Scaffold `lepenseur` as a full CLI/PyPI AgentCulture sibling

**Date:** 2026-05-22
**Issue:** [agentculture/lepenseur#1](https://github.com/agentculture/lepenseur/issues/1)
**Status:** Design — approved for spec write

## 1. Goal

Take `lepenseur` from its current early scaffold (`README.md`, `LICENSE`,
`.gitignore`, `CLAUDE.md`) to a **full AgentCulture CLI/PyPI sibling**, matching
the sibling pattern and satisfying issue #1's acceptance criteria. `lepenseur`
("le penseur") is the local **thinking** agent of the Culture mesh: it reasons,
plans, and analyzes, and acts **only through writing** (post/reply on Culture
chat, create files) — never code execution or side effects.

Runtime: served by a **locally-hosted vLLM model over the `acp` backend** (not a
Claude-backed agent), model
`vllm-local/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`.

## 2. Approach

**Copy the twin, then layer deltas.** `lecodeur` ("le codeur", the local coding
agent) is lepenseur's already-scaffolded near-twin: same parallel naming, same
`acp`/vLLM runtime shape, same top-level package layout, both CI pipelines, the
exact same six vendored skills, the same lint configs and
`skills.local.yaml.example`. Its own `AGENTS.md` already names lepenseur as its
closest sibling. lecodeur was itself built on the afi-cli pattern, so copying it
inherits the afi CLI shape via an already-adapted sibling rather than
re-running `afi cli cite` from scratch.

`afi`'s role in this build is therefore the **quality auditor**, not the
scaffolder: the seven-bundle agent-first rubric (`afi cli doctor . --strict`)
becomes a CI gate.

The three pillars from the issue map as:

- **afi** — CLI shape inherited from lecodeur (afi-pattern) + `afi cli doctor`
  rubric gate.
- **pipelines** — `tests.yml` + `publish.yml` copied from lecodeur, renamed.
- **quality** — lint + pytest/coverage + the `afi cli doctor . --strict` gate.

### Build sequence

1. Copy lecodeur's tracked tree into `lepenseur/`, excluding lecodeur's
   `README.md`/`LICENSE`/`.gitignore`/`CLAUDE.md` (lepenseur already has its
   own) and lecodeur's `CHANGELOG.md` (start fresh).
2. Rename `lecodeur` → `lepenseur` across package dir, imports, `pyproject.toml`,
   workflows, `culture.yaml`, and skill prose.
3. Set packaging to bare `lepenseur` (dist == package == script), matching the
   lecodeur twin (see §6).
4. Add the `overview` and `doctor` verbs + a thin `cli overview` (§4). `doctor`
   ships as a rubric-shaped **stub** (semantics deferred — see §4, §12).
5. Add the `afi cli doctor . --strict` gate and `afi-cli` dev-dep (§7).
6. Rewrite identity content: `AGENTS.md`, `culture.yaml` system prompt, `explain`
   catalog, `learn` text, `README.md` (§4, §8).
7. Reframe the six `SKILL.md` files (`lecodeur`/`steward` → `lepenseur`) and add
   provenance notes pointing to steward as canonical upstream (§9).
8. Start `CHANGELOG.md` fresh; verify against acceptance criteria (§11).

## 3. Target tree

```text
lepenseur/
├── __init__.py                 # __version__ via importlib.metadata("lepenseur")
├── __main__.py                 # python -m lepenseur → cli.main
└── cli/
    ├── __init__.py             # argparse parser + _dispatch (registers verbs + cli noun)
    ├── _errors.py              # LepenseurError + EXIT_SUCCESS/USER/ENV
    ├── _output.py              # emit_result / emit_error / emit_diagnostic (stdout/stderr split)
    └── _commands/
        ├── __init__.py
        ├── whoami.py           # identity probe — reads culture.yaml
        ├── learn.py            # self-teaching prompt (≥200 chars; rubric bundle 2)
        ├── explain.py          # catalog lookup (rubric bundle 5)
        ├── overview.py         # descriptive snapshot + shared section builder (rubric bundle 6)
        ├── cli.py              # `cli` noun → `cli overview` (rubric overview_cli_noun_exists)
        └── doctor.py           # self-diagnosis STUB (rubric bundle 7; semantics deferred)
lepenseur/explain/
├── __init__.py                 # resolve() + known_paths() over ENTRIES
└── catalog.py                  # ENTRIES dict — lepenseur's command docs
tests/
├── __init__.py
└── test_cli_*.py               # per-verb smoke + json + error tests
.github/workflows/
├── tests.yml                   # test + lint(+afi cli doctor) + version-check
└── publish.yml                 # PyPI (main) / TestPyPI (PR) via Trusted Publishing
.claude/skills/{cicd,communicate,version-bump,run-tests,sonarclaude,doc-test-alignment}/
.claude/skills.local.yaml.example
pyproject.toml                  # hatchling, py≥3.12, dist lepenseur (bare)
AGENTS.md                       # runtime system prompt (thinker)
culture.yaml                    # backend: acp + Nemotron model + mirrored prompt
CHANGELOG.md                    # Keep-a-Changelog (fresh)
.flake8  .markdownlint-cli2.yaml
README.md  LICENSE  .gitignore  CLAUDE.md   # already present
docs/superpowers/specs/2026-05-22-scaffold-cli-sibling-design.md
```

## 4. CLI verbs (the agent-first quartet + whoami)

All verbs are **read-only** — consistent with lepenseur's "thinker, not actor"
constraint. The dry-run/`--apply` convention is documented for any future write
verb but no write verb ships now.

| Verb | Origin | Behavior | `--json` shape |
|------|--------|----------|----------------|
| `whoami` | from lecodeur | Read `culture.yaml`; report suffix, backend, model | identity object |
| `learn` | from lecodeur (afi template) | Self-teaching prompt; ≥200 chars; names purpose/commands/exit/`--json`/`explain` | `{tool, version, purpose, commands, exit_codes, json_support, explain_pointer}` |
| `explain <path>` | from lecodeur (afi template) | Markdown catalog lookup; unknown path → `LepenseurError` | `{path, markdown}` |
| `overview` | **new**, lepenseur-bespoke | Read-only descriptive snapshot of the agent (identity, verbs, act surface); graceful (exit 0) on a bogus path arg | `{subject, sections}` |
| `cli overview` | **new**, thin | `cli` noun whose `overview` describes the CLI surface (verbs, `--json`, exit codes). Reuses overview's section builder | `{subject, sections}` |
| `doctor` | **new**, **stub** | Rubric-shaped self-diagnosis. Ships as a stub returning a single passing check; real semantics deferred (see §12) | `{healthy: bool, checks: [{id, passed, severity, message, remediation}]}` |

The `explain` catalog (`lepenseur/explain/catalog.py`) must include a
`("backend",)` entry (describing the `acp`/vLLM-local runtime) so that
`lepenseur explain backend` — an acceptance-criteria invocation — resolves.
The catalog also covers the root, `learn`, `explain`, `whoami`, `overview`,
and `doctor`.

### Why `overview` + `cli overview` + `doctor` (rubric-forced surface)

The seven-bundle rubric (`afi cli doctor`) makes three **error-severity**
demands — error-severity checks fail the gate *even without* `--strict`:

- **bundle 6 (`overview`)**: `overview` exits 0 non-empty; `overview --json`
  carries keys `subject` + `sections`; `overview <bogus-path>` exits 0
  (graceful); **and `cli overview` exits 0 non-empty** (`overview_cli_noun_exists`
  is unconditional — hence the thin `cli` noun).
- **bundle 7 (`doctor`)**: `doctor` non-empty stdout; `doctor --json` →
  `{healthy: bool, checks: [...]}`; each check has `id`/`passed`/`severity`/
  `message`; failed checks carry non-empty `remediation`.

`doctor` is shipped as an honest **stub**: its real meaning for a *non-doer*
(a thinker that never executes) is undefined — it might later check the vLLM
endpoint is reachable, or that `culture.yaml`/`AGENTS.md` are coherent. For now
it returns one trivially-passing check (`healthy: true`) so the gate is green,
with the open question tracked in §12. `overview` is implemented for real (a
thinker describing itself is well-defined).

### Error / output contract (inherited, renamed)

- `cli/_errors.py`: `LepenseurError` dataclass carrying `code`, `message`,
  `remediation`; exit codes `EXIT_SUCCESS=0`, `EXIT_USER_ERROR=1`,
  `EXIT_ENV_ERROR=2`.
- `cli/_output.py`: results to **stdout**, errors/diagnostics to **stderr**,
  never mixed. `--json` mode emits structured JSON; errors in JSON mode emit
  `{code, message, remediation}` to stderr.
- `cli/__init__.py`: `_ArgumentParser` override routes argparse errors through
  the structured emitter; `_dispatch` catches `LepenseurError` and wraps unknown
  exceptions so no traceback leaks.

## 5. Runtime files

- **`AGENTS.md`** — the running agent's system prompt. Mirror of lecodeur's
  framing, inverted to the thinker role: lepenseur reasons/plans/analyzes;
  lecodeur builds; daria observes. Act surface is explicitly **post to Culture
  chat, reply on Culture chat, create files** — nothing else. Style:
  observation → interpretation → next step; distinguish facts, inferences,
  recommendations; ask one focused question when ambiguous.
- **`culture.yaml`** — one agent entry: `suffix: lepenseur`, `backend: acp`,
  `model: vllm-local/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`, an inline
  `system_prompt:` mirroring `AGENTS.md`, and `acp_command: [opencode, acp]`.

## 6. PyPI / packaging contract

**Bare `lepenseur`** — dist == package == script — matching the lecodeur twin.

- `pyproject.toml` `[project] name = "lepenseur"`.
- `[project.scripts] lepenseur = "lepenseur.cli:main"`.
- `[tool.hatch.build.targets.wheel] packages = ["lepenseur"]`.
- `lepenseur/__init__.py`: `__version__ = importlib.metadata.version("lepenseur")`.
- Version single source of truth in `pyproject.toml`; no separate literal.
- Coverage `source = ["lepenseur"]`, `fail_under = 60` (lecodeur's bar).
- `dev` dependency group from lecodeur **plus `afi-cli`** (for the doctor gate).

> **Doc alignment required:** the committed `CLAUDE.md` and issue #1 both say
> dist `lepenseur-cli` / `importlib.metadata("lepenseur-cli")`. This scaffold
> uses bare `lepenseur` instead, so `CLAUDE.md`'s "Target project shape" and
> "Build / test / publish" sections must be updated to drop the `-cli` suffix as
> part of this work. The issue is steward-owned and is left as-is (the PR
> description will note the intentional deviation).

## 7. Pipelines

Copied from lecodeur, renamed `lecodeur`→`lepenseur`, `paths:` filter set to
`pyproject.toml` and `lepenseur/**`.

- **`tests.yml`** — jobs:
  - `test`: `uv sync` → `uv run pytest -n auto --cov=lepenseur …`; optional
    SonarCloud scan gated on `SONAR_TOKEN`.
  - `lint`: `black --check`, `isort --check-only`, `flake8`, `bandit -r
    lepenseur`, `markdownlint-cli2`, **plus a new step `uv run afi cli doctor
    . --strict`** (the seven-bundle agent-first rubric gate; blocking).
  - `version-check` (PR-only): fails if `pyproject.toml` version equals main's
    (AgentCulture every-PR-bumps rule).
- **`publish.yml`** — push-to-`main` builds with `uv build` and publishes
  `lepenseur` to PyPI via Trusted Publishing (no API tokens); PRs publish a
  `.dev<run_number>` to TestPyPI; fork PRs skipped (no OIDC).

## 8. Lint / changelog / config

- `.flake8`, `.markdownlint-cli2.yaml` — copied from lecodeur (repo-local; no
  home-dir configs).
- `CHANGELOG.md` — fresh Keep-a-Changelog, first entry documents the scaffold.
- `.claude/skills.local.yaml.example` — copied from lecodeur (committed);
  `.claude/skills.local.yaml` stays git-ignored.

## 9. Vendored skills (cite-don't-import)

Six skills, copied from lecodeur (whose set already matches issue #1 exactly):
`cicd`, `communicate`, `version-bump`, `run-tests`, `sonarclaude`,
`doc-test-alignment`. Per-copy work:

- **Scripts copied verbatim.** They depend on installed binaries (`agex` 0.17,
  `agtag` 0.2.1, `gh`, `jq`, `culture` 12.1.6 — all present), not on paths
  outside the repo, so they are portable as-is.
- **Signing identity is resolved dynamically**, not hardcoded: `communicate`'s
  `agtag` and `cicd`'s `_resolve-nick.sh` both read the signing nick from the
  repo's `culture.yaml` → posts auto-sign **`- lepenseur (Claude)`**. No
  signature literal to edit.
- **`SKILL.md` prose reframed** `lecodeur`/`steward` → `lepenseur`, with a
  one-line provenance note crediting **steward** as the canonical upstream
  (`docs/skill-sources.md`).

## 10. Mutation safety

The initial verb set is entirely read-only. The dry-run-by-default / `--apply`
convention is documented (in `CLAUDE.md` and `AGENTS.md`) for any future write
verb, but no write verb is introduced in this scaffold.

## 11. Acceptance criteria (from issue #1)

- [ ] `uv sync && uv run pytest -n auto -v` passes locally.
- [ ] `uv run lepenseur --version`, `lepenseur whoami`, `lepenseur whoami --json`,
      `lepenseur learn`, `lepenseur explain backend`, `lepenseur overview`,
      `lepenseur overview --json`, `lepenseur cli overview`, `lepenseur doctor`,
      `lepenseur doctor --json` all work.
- [ ] `black --check`, `isort --check-only`, `flake8`, `bandit -r lepenseur`
      clean.
- [ ] `uv run afi cli doctor . --strict` exits 0 (all seven bundles pass).
- [ ] `tests.yml` + `publish.yml` present; `version-check` job in place.
- [ ] `AGENTS.md` (runtime) and `CLAUDE.md` (dev) both present; `culture.yaml`
      declares `backend: acp` + the `vllm-local/...` Nemotron model.
- [ ] All six skills under `.claude/skills/`, each with `SKILL.md` + `scripts/`;
      `.claude/skills.local.yaml.example` committed.
- [ ] `.flake8` + `.markdownlint-cli2.yaml` present; `CHANGELOG.md` started.
- [ ] Version bumped vs main (`version-check` passes).

## 12. Out of scope / follow-ups

- **Define `doctor` semantics for a thinking ("non-doer") agent.** This scaffold
  ships `doctor` as a rubric-shaped stub. The real checks are an open design
  question — candidates: vLLM endpoint reachability, `culture.yaml`/`AGENTS.md`
  coherence, model-string validity. File a follow-up issue on lepenseur to track
  it; the stub's docstring points there.
- **steward PR** adding `lepenseur` to `docs/skill-sources.md` "Downstream
  copies" column (steward is never edited from here).
- **PyPI/TestPyPI Trusted Publishing setup** for the `lepenseur` project is a
  one-time external configuration on the PyPI side (OIDC publisher); the
  workflow ships ready but first publish requires that registration.
- No write/mutation verbs; no MCP/HTTP surface (CLI only).

## 13. Testing strategy

- Per-verb smoke tests in `tests/test_cli_*.py`: each verb exits 0 on the happy
  path, emits non-empty stdout, and `--json` parses to the documented shape.
- `overview` tests assert `subject`/`sections` JSON keys and **exit 0 on a bogus
  path arg** (the rubric's graceful-fallback contract); `cli overview` exits 0
  non-empty; `doctor --json` returns `{healthy, checks}` with each check
  carrying `id`/`passed`/`severity`/`message`.
- Error-path tests: bad `explain` path → exit 1 + `LepenseurError` on stderr;
  unknown subcommand → structured argparse error.
- `whoami` reads a fixture/temp `culture.yaml`.
- The `afi cli doctor . --strict` gate is the black-box integration check across
  all seven rubric bundles; local tests cover the white-box behavior.
- Coverage `fail_under = 60`.
