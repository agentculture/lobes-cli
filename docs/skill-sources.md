# Skill sources

This is lobes' provenance ledger for the skills under `.claude/skills/`.
The policy is **cite-don't-import**: every vendored skill is *copied* into this
repo and is **owned by lobes** — it may diverge from its upstream, and this
repo's copy is authoritative for this repo. Nothing here symlinks to or
runtime-depends on a sibling checkout.

"Citation path" is where the copy was taken from when a sibling is checked out
alongside this repo (the shared-workspace layout). "Origin" is the repo that
*authors and maintains* the skill upstream. The two differ for the devague skills:
those are authored in `agentculture/devague` and *re-broadcast* through
`agentculture/guildmaster`, so guildmaster is the citation point even though
devague is the author.

## devague operator skills — origin `agentculture/devague`, via guildmaster

The idea→spec→plan→implementation→evidence→closure operator chain for the
deterministic `devague` CLI — **eight legs**, in flow order. Each leg hands off
to the next; the `SKILL.md` descriptions cross-reference one another. They carry
`type: command` in their frontmatter, which is load-bearing on the culture/agex
backend (a `SKILL.md` without `type:` is silently skipped when the repo declares
an agent in `culture.yaml`).

Two shapes ship upstream: **CLI-driving** skills (`think`, `spec-to-plan`,
`assign-to-workforce`) carry a `scripts/<name>.sh` portable CLI resolver;
**method-only** skills (`scope`, `challenge`, `deviate`, `validate-delivery`,
`summarize-delivery`) are `SKILL.md` alone and invoke the `devague` CLI directly.

| Skill | Shape | Citation path | Origin | Notes |
|-------|-------|---------------|--------|-------|
| `scope` | method-only | `../guildmaster/.claude/skills/scope/` | `agentculture/devague` | idea→explored-scope leg (optional opener). Verbatim copy — no divergence. |
| `think` | CLI-driving | `../guildmaster/.claude/skills/think/` | `agentculture/devague` | idea→spec leg. Verbatim copy, incl. `type: command` — no divergence. |
| `challenge` | method-only | `../guildmaster/.claude/skills/challenge/` | `agentculture/devague` | blind-spot pass between spec export and `plan new`. Verbatim copy — no divergence. |
| `spec-to-plan` | CLI-driving | `../guildmaster/.claude/skills/spec-to-plan/` | `agentculture/devague` | spec→plan leg (drives `devague plan`). Verbatim copy — no divergence. |
| `assign-to-workforce` | CLI-driving | `../guildmaster/.claude/skills/assign-to-workforce/` | `agentculture/devague` | plan→parallel implementation leg. Verbatim copy — no divergence. |
| `deviate` | method-only | `../guildmaster/.claude/skills/deviate/` | `agentculture/devague` | execution-time: records human-approved departures from the confirmed plan. Verbatim copy — no divergence. |
| `validate-delivery` | method-only | `../guildmaster/.claude/skills/validate-delivery/` | `agentculture/devague` | execution→evidence leg: runs the plan's behavioral tests agent-side, files evidence + behavioral deltas. Added 2026-09-02. Verbatim copy — no divergence. |
| `summarize-delivery` | method-only | `../guildmaster/.claude/skills/summarize-delivery/` | `agentculture/devague` | closure leg: the committed accountability artifact. Verbatim copy — no divergence. |

Runtime dependency (all eight): the `devague` CLI (`uv tool install devague`).
The three CLI-driving wrappers resolve it portably — an installed `devague` on
`PATH`, falling back to `uv run devague` inside a devague checkout — so no
dependency is added to `pyproject.toml`.

Refreshing these copies is a **wholesale re-vendor**, not a merge: they have no
lobes-local divergence, so the update is `devague learn skills:<name>` for the
source URL and a verbatim overwrite. `devague learn` is the authority on how
many legs exist and what each one does; if it disagrees with the table above,
`devague learn` is right and this table is stale.

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
| `model-runner` | — | lobes | Not vendored. Thin shim that `exec`s the `lobes` CLI (this repo's `lobes` package), with `model` as a deprecated alias. |
