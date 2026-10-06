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
4. **Start or stop one role with `lobes up <role> [--down] --apply`** (this
   includes `innereye` and `gateway`). For any other compose action, run
   compose only through `scripts/lobes-compose.sh`. It uses the
   deployment dir and its full `-f` chain, refuses the template folder, and is
   dry-run until `--apply`. Target ONE service with `up -d --no-deps <svc>`.
5. **Verify through the gateway**, not just `docker ps`: the gateway must
   resolve the service (`docker exec model-gear-gateway getent hosts <svc>`)
   and a real request must succeed.
6. **Then** carry the template side as a PR (template + test + version bump).

## Recipe: serve innereye (ComfyUI) with its web UI on the network

Follow these steps in order. You don't need to read any code or `.env` first.
The `lobes` CLI does the swap. Don't do it by hand.

On the DGX Spark, `innereye` and `cortex` **can't run together**. They share
one unified memory pool, and the card profile declares them exclusive. Turning
innereye on therefore turns cortex off on this box. `model=cortex` keeps
working because the mesh serves it from another member.

1. **Web UI on the network:** `grep '^INNEREYE_UI_PORT' ~/.lobes/.env` must
   show `INNEREYE_UI_PORT=0.0.0.0:8188`. If it doesn't, back up `.env` and
   set it. A bare `8188` binds loopback only. ComfyUI has **no login**:
   anyone on the network can use the GPU and read every past prompt and
   output, so set this only when the operator asks for network access.
2. **Look at the plan:** `lobes up innereye --replace`. It's a dry run that
   lists every step: stop cortex, the `.env` keys it writes (it backs the
   file up first), start `comfyui`, recreate the gateway, and a memory line.
3. **Do it:** `lobes up innereye --replace --apply`.
   - When nothing has to stop, it refuses `--replace`. Use
     `lobes up innereye --apply`.
   - **`not enough memory`:** stop and tell the operator what it printed.
     After a `--replace` it has already restarted cortex, so nothing changed.
     Pass `--override-memory` only if the operator says so.
4. **Verify:**

   ```bash
   docker ps --filter name=model-gear-comfyui --format '{{.Status}} {{.Ports}}'
   #   want: Up … (healthy) 0.0.0.0:8188->8188/tcp
   for ip in $(hostname -I); do curl -s -o /dev/null -m 3 -w "$ip %{http_code}\n" http://$ip:8188/; done
   #   want: 200 on the LAN and tailnet addresses (Docker's 172.x bridges don't matter)
   curl -s localhost:8001/capabilities | python3 -c 'import json,sys; r=json.load(sys.stdin)["innereye"]; print(r["feasible"], r["ready"])'
   #   want: True True   (8001 = this box's gateway port, VLLM_PORT in .env)
   ```

**Switch back to cortex:** `lobes up cortex --replace --apply`. Cortex needs a
few minutes to load before it reports ready.

**If the port is missing from `docker ps`:** the compose files don't read
`INNEREYE_UI_PORT`. Check with
`grep -n INNEREYE_UI_PORT ~/.lobes/docker-compose.{override,shape}.yml`. On a
box whose compose files are rendered, not hand-kept, re-render with
`lobes init --shape <its shape> --apply`. On a hand-kept box (each file's
header says so), stop and ask. Don't add `ports:` to a template.

Never `docker rm -f` a container you did not create, and never add a `ports:`
key to a packaged template to fix one box. The full reasoning, the ComfyUI
specifics and the 2026-09-18 incident are in
[`docs/operating-a-deployment.md`](../../../docs/operating-a-deployment.md).
