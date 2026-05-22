"""Markdown catalog for ``lepenseur explain <path>``.

Each entry is verbatim markdown. Keys are topic-path tuples. The empty tuple
and ``("lepenseur",)`` both resolve to the root entry (aliased).

Keep bodies self-contained — an agent reading a single entry should get enough
context without chaining reads.
"""

from __future__ import annotations

_ROOT = """\
# lepenseur

lepenseur ("le codeur" — *the coder*) is the **local coding agent** of the
Culture mesh: a long-lived resident that implements, edits, and tests code.

## The matched pair: thinker + coder

lepenseur is the *doer* of a matched pair. Its closest sibling is
**lepenseur** ("le penseur" — *the thinker*), which reasons, plans, and
analyzes but only acts through writing (chat + files). lepenseur plans;
lepenseur executes. The next-closest sibling is **daria** (awareness), which
observes the mesh and surfaces drift. Together: daria notices, lepenseur
reasons, lepenseur builds.

## Verbs

- `lepenseur whoami` — smallest identity probe: nick, version, backend, served
  model (read from `culture.yaml`).
- `lepenseur learn` — structured self-teaching prompt.
- `lepenseur explain <path>` — markdown docs for a topic.

## Mutation safety

Any verb that writes defaults to **dry-run**; pass `--apply` to commit. Agents
call CLIs in loops, so safe-by-default is mandatory. The verbs above are
read-only.

## Exit-code policy

- `0` success
- `1` user-input error (bad flag, bad path, missing arg)
- `2` environment / setup error (tool not installed, unreadable file)
- `3+` reserved

## See also

- `lepenseur explain backend`
- `lepenseur explain whoami`
- `lepenseur explain learn`
- `lepenseur explain explain`
"""

_BACKEND = """\
# lepenseur backend

lepenseur is **not** a Claude-backed agent. At runtime it is served by a
**locally-hosted vLLM code model over the `acp` backend** — `daria` is the
worked example of this runtime shape.

## Declaration (`culture.yaml`)

```yaml
agents:
- suffix: lepenseur
  backend: acp
  model: vllm-local/Qwen/Qwen3-Coder-Next
  acp_command: [opencode, acp]
```

## Served model

`Qwen/Qwen3-Coder-Next` — an 80B-total / 3B-active MoE code model. The official
weights are BF16 (~160GB) and do **not** fit a 128GB DGX Spark, so the local
deployment serves a **quantized variant** (FP8 ≈ ~80GB, or a community quant).
The `model:` string names the family; the exact quantized repo is pinned at
serve time.

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

Prints a structured self-teaching prompt: purpose, the command map, mutation
safety, the `--json` contract, and the exit-code policy. Enough shape for an
agent to author its own usage skill without scraping `--help`. Supports
`--json`.
"""

_EXPLAIN = """\
# lepenseur explain

Resolves a topic path against the markdown catalog and prints the body. With no
path it returns the root overview (same as `lepenseur explain lepenseur`). Unknown
paths exit `1` with a `hint:` pointing back at the root. Supports `--json`,
which wraps the markdown as `{"path": [...], "markdown": "..."}`.
"""

ENTRIES: dict[tuple[str, ...], str] = {
    (): _ROOT,
    ("lepenseur",): _ROOT,
    ("backend",): _BACKEND,
    ("whoami",): _WHOAMI,
    ("learn",): _LEARN,
    ("explain",): _EXPLAIN,
}
