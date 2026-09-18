# QWEN.md

Guidance for Qwen Code (and any other coding agent) working in this repository.

- **Read `CLAUDE.md` first.** It is the maintainer guide for this repo and
  applies to you too: tests, lint, version bump on every PR, PR workflow.
- **`AGENTS.md` is not for you.** It is the system prompt of the deployed
  `lobes` agent, a thinker that doesn't run tools. Don't take its "you do not
  execute code" rules as instructions for your session.

## Before you touch a running box

**`lobes/templates/**` holds packaged templates. The running deployment is
`~/.lobes` (or `$LOBES_DIR`).** Never run `docker compose` inside this repo,
never `docker rm -f` a container you did not create, and never add a `ports:`
key to a template to fix one box.

For any "it's on docker, fix it / rebuild it / make it reachable" request, use
the `lobes-deploy` skill (`.qwen/skills/lobes-deploy`) and its wrapper
`scripts/lobes-compose.sh`. The full procedure and why it exists are in
`docs/operating-a-deployment.md`.
