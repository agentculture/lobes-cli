---
name: lobes-deploy
type: command
description: >
  Change what a LIVE lobes box runs — rebuild an image, recreate one container,
  publish a port, change a knob — without breaking the deployment. Use when a
  served lane (ComfyUI/innereye, a vLLM lane, the gateway, audio) is broken or
  needs a change on the running box, or the user says "redeploy", "rebuild the
  container", "restart comfyui", "make it reachable from the network", "it's on
  docker, fix it". The one rule: the repo's lobes/templates/** are PACKAGED
  TEMPLATES, not the deployment. The deployment is ~/.lobes (or $LOBES_DIR).
  Never run `docker compose` inside the repo.
---

# lobes-deploy

A lobes box has **two copies** of every compose file and Dockerfile:

| | Where | What it is | How it changes |
|---|---|---|---|
| **Template** | `lobes/templates/**` in this repo | What the NEXT `lobes init` writes, on every box, for every user | Edit + test + version bump + PR |
| **Deployment** | `~/.lobes` (or `$LOBES_DIR`) | What THIS box is running right now | Edit in place, back up first |

A fix usually needs **both**: the template, so the next box gets it, and the
deployment, so this box gets it now. Fixing only the template changes nothing
that's running. Running compose in the template folder creates a second,
broken deployment.

## When to use

- A served lane fails at runtime and the fix is in its image or compose service.
- The user wants a container rebuilt, recreated, or exposed differently.
- Before you run any `docker compose`, `docker rm`, or `docker stop` on a lobes box.

## How

1. **Find the deployment**, never assume the repo: `lobes fleet files --json`
   prints its dir and file chain. Read its `.env`, `docker-compose.override.yml`
   and `docker-compose.shape.yml` before editing. On the Spark those files are
   **hand-kept** and must not be re-scaffolded (`lobes init --force` would
   overwrite them).
2. **Look for a knob first.** Many "make it do X" asks are already an `.env`
   key (e.g. `INNEREYE_UI_PORT` publishes ComfyUI's UI). Change the knob, not
   the template.
3. **Back up** every deployment file you touch:
   `cp FILE FILE.bak-$(date +%Y%m%d-%H%M%S)-<why>`.
4. **Run compose only through** `scripts/lobes-compose.sh`. It uses the
   deployment dir and its full `-f` chain, refuses the template folder, and is
   dry-run until `--apply`. Target ONE service with `up -d --no-deps <svc>`.
5. **Verify through the gateway**, not just `docker ps`: the gateway must
   resolve the service (`docker exec model-gear-gateway getent hosts <svc>`)
   and a real request must succeed.
6. **Then** carry the template side as a PR (template + test + version bump).

Never `docker rm -f` a container you did not create, and never add a `ports:`
key to a packaged template to fix one box. The full reasoning, the ComfyUI
specifics and the 2026-09-18 incident are in
[`docs/operating-a-deployment.md`](../../../docs/operating-a-deployment.md).
