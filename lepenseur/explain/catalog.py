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
  model: vllm-local/nvidia/Qwen3-32B-NVFP4
  acp_command: [opencode, acp]
```

## Served model

`nvidia/Qwen3-32B-NVFP4` — a 32B dense reasoning model in NVFP4, with a
32K-token context (extendable to ~131K via YaRN), running on DGX Spark. It has a
thinking mode and emits a reasoning trace before its answer — which is exactly
why it suits a deep thinker.

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
