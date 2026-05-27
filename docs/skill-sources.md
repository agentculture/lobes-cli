# Skill sources

This is model-gear's provenance ledger for the skills under `.claude/skills/`.
The policy is **cite-don't-import**: every vendored skill is *copied* into this
repo and is **owned by model-gear** — it may diverge from its upstream, and this
repo's copy is authoritative for this repo. Nothing here symlinks to or
runtime-depends on a sibling checkout.

"Citation path" is where the copy was taken from when a sibling is checked out
alongside this repo (the shared-workspace layout). "Origin" is the repo that
*authors and maintains* the skill upstream. The two differ for the devague trio:
those skills are authored in `agentculture/devague` and *re-broadcast* through
`agentculture/guildmaster`, so guildmaster is the citation point even though
devague is the author.

## devague workflow trio — origin `agentculture/devague`, via guildmaster

The idea→spec→plan→implementation operator chain for the deterministic `devague`
CLI. Each leg hands off to the next; the `SKILL.md` descriptions cross-reference
one another. They carry `type: command` in their frontmatter, which is
load-bearing on the culture/agex backend (a `SKILL.md` without `type:` is
silently skipped when the repo declares an agent in `culture.yaml`).

| Skill | Citation path | Origin | Notes |
|-------|---------------|--------|-------|
| `think` | `../guildmaster/.claude/skills/think/` | `agentculture/devague` | idea→spec leg. Verbatim copy, incl. `type: command` — no divergence. |
| `spec-to-plan` | `../guildmaster/.claude/skills/spec-to-plan/` | `agentculture/devague` | spec→plan leg (drives `devague plan`). Verbatim copy — no divergence. |
| `assign-to-workforce` | `../guildmaster/.claude/skills/assign-to-workforce/` | `agentculture/devague` | plan→parallel implementation leg. Verbatim copy — no divergence. |

Runtime dependency (all three): the `devague` CLI (`uv tool install devague`).
The wrappers resolve it portably — an installed `devague` on `PATH`, falling
back to `uv run devague` inside a devague checkout — so no dependency is added to
`pyproject.toml`.

## steward skills — origin `agentculture/steward`

Vendored from steward, the canonical upstream for these six. steward owns the
sibling-pattern contract and files issues on this repo but never edits it; copies
here may diverge.

| Skill | Citation path | Origin | Notes |
|-------|---------------|--------|-------|
| `cicd` | `../steward/.claude/skills/cicd/` | `agentculture/steward` | CI/CD lane (layered on `agex pr`). |
| `communicate` | `../steward/.claude/skills/communicate/` | `agentculture/steward` | Cross-repo + mesh communication. |
| `version-bump` | `../steward/.claude/skills/version-bump/` | `agentculture/steward` | Semver bump + CHANGELOG entry. |
| `run-tests` | `../steward/.claude/skills/run-tests/` | `agentculture/steward` | pytest with parallelism + coverage. |
| `sonarclaude` | `../steward/.claude/skills/sonarclaude/` | `agentculture/steward` | SonarCloud quality-gate queries. |
| `doc-test-alignment` | `../steward/.claude/skills/doc-test-alignment/` | `agentculture/steward` | Doc↔code/test alignment check (stub). |

## Local to this repo

| Skill | Citation path | Origin | Notes |
|-------|---------------|--------|-------|
| `model-runner` | — | model-gear | Not vendored. Thin shim that `exec`s the `model` CLI (this repo's `model_gear` package). |
