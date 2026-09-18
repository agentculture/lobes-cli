# Operating a live deployment: templates are not the deployment

This page is for anyone, human or agent, about to change what a running lobes
box does: rebuild an image, recreate a container, publish a port, change a
knob. The skill that drives it is
[`lobes-deploy`](../.claude/skills/lobes-deploy/SKILL.md). Its wrapper,
`scripts/lobes-compose.sh`, enforces the one rule below.

## The one rule

**`lobes/templates/**` in this repo is a set of packaged templates. The
deployment is `~/.lobes`, or `$LOBES_DIR`.** They look alike (the same
`docker-compose.yml`, the same `Dockerfile.*`), but they do different jobs:

| | Template (`lobes/templates/**`) | Deployment (`~/.lobes`) |
|---|---|---|
| What it is | What the next `lobes init` writes on every box | What this box is running now |
| Who it affects | Every user of the next release | This box only |
| How it changes | Edit, test, version bump, PR, publish | Edit in place, after a backup |
| `docker compose` here? | **Never** | Yes, through `lobes-compose.sh` or a `lobes` verb |

`lobes fleet files --json` prints the deployment dir and the exact `-f` chain
the CLI uses for it. Start there, not with a `grep` of the repo.

A runtime fix usually needs both halves: the **deployment**, so this box works
now, and the **template**, so the next box doesn't hit the same problem. Do
the deployment half first and prove it with a real request. Then carry the
template half as a PR.

## Why running compose in the template folder breaks things

`docker compose` names a project after the folder it runs in. Run it in
`lobes/templates/fleet/` and you get a **second project called `fleet`**
beside the real one (called `lobes`, after `~/.lobes`). The second project:

- **Takes the container name.** Every service pins `container_name:
  model-gear-<svc>`, so the new container can only start once the real one has
  been removed. The real deployment loses the service without noticing.
- **Sits on a different network.** The new container joins `fleet_default`.
  The gateway is on the `lobes` network and can no longer resolve the service,
  so every gateway-fronted request to it fails while `docker ps` shows it
  healthy.
- **Ignores the box's own configuration.** The deployment's `.env`,
  `docker-compose.shape.yml` and `docker-compose.override.yml` are not in the
  template folder, so every knob falls back to its template default.
- **Writes files into the repo.** It bind-mounts `./mg-logwrap.sh` and
  `./logs`, which a scaffold creates but the template folder does not have. The
  run leaves a root-owned `logs/` folder and a copied script inside
  `lobes/templates/`. The copied script would ship in the wheel if committed.

## Change a knob before you change a file

Many requests of the form "make the service do X" are already an `.env` key.
For example, publishing ComfyUI's web UI is `INNEREYE_UI_PORT`, not a `ports:`
line in the template (see
[`comfyui-innereye.md`](comfyui-innereye.md#the-third-exposure--innereye_ui_port-opt-in-default-off)).
Search `env.example` and the per-lane doc for a knob before editing anything.
A `ports:` line added to a packaged template publishes that port on **every**
box that renders it.

## Hand-kept boxes

Some boxes carry deployment files that are **hand-maintained** rather than
rendered. The DGX Spark is the main case: its `docker-compose.shape.yml` and
`docker-compose.override.yml` are hand-edited, and each carries a header
comment saying so. On such a box:

- Do **not** run `lobes init --shape … --apply --force` to pick up a change.
  It overwrites those files from the templates with no backup.
- Knobs that `lobes init` would render into the shape file (like
  `INNEREYE_UI_PORT`) take effect only if the hand-kept file already reads
  them. On the Spark, `docker-compose.override.yml`'s `comfyui` block reads
  `${INNEREYE_UI_PORT}` directly, so changing `.env` and recreating the one
  container is enough.
- Copy a changed template file (for example `Dockerfile.comfyui`) into the
  deployment by hand, after backing up the old one.

## The procedure

1. **Locate:** `lobes fleet files --json`. Read the deployment's `.env`,
   override and shape files for the service you're changing.
2. **Back up** each file you'll touch:
   `cp FILE FILE.bak-$(date +%Y%m%d-%H%M%S)-<reason>`.
3. **Change** the knob, or copy the fixed template file into the deployment.
4. **Rebuild and recreate one service** through the wrapper:

   ```bash
   S=.claude/skills/lobes-deploy/scripts/lobes-compose.sh
   $S --apply build comfyui
   $S --apply up -d --no-deps comfyui
   ```

   `--no-deps` is required: without it, compose walks `depends_on` and
   recreates other lanes (issue #222). A service behind a compose profile
   (`comfyui` is behind `innereye`) needs `--profile innereye` for `stop` or
   `down` to see it. Where the installed `lobes up <role>` knows the role,
   that verb is equivalent.
5. **Verify through the gateway**, not only `docker ps`:

   ```bash
   docker exec model-gear-gateway getent hosts comfyui   # the gateway resolves it
   ```

   Then make one real request through the gateway (for ComfyUI, a small
   `POST /v1/render`, polled at `GET /v1/render/jobs/{id}` until `state` is
   `completed`).
6. **Carry the template half** as a PR: the template change, a test that pins
   it, and a version bump.

Never `docker rm -f` a container you didn't create. If a name conflict blocks
you, find the owning project first:
`docker inspect <name> --format '{{index .Config.Labels "com.docker.compose.project.working_dir"}}'`.

## The 2026-09-18 incident

On the DGX Spark, ComfyUI renders failed at their first sampler step with
`RuntimeError: Failed to find C compiler`: Triton builds a small C launcher
the first time it runs a kernel, and `Dockerfile.comfyui` shipped no compiler.
An agent asked to fix it:

1. correctly added `build-essential` to the **template** Dockerfile;
2. ran `docker compose up -d --build comfyui` **inside
   `lobes/templates/fleet/`**, then `docker rm -f model-gear-comfyui` to clear
   the name conflict, which removed the real deployment's container;
3. copied `mg-logwrap.sh` into the template folder to satisfy the bind mount;
4. when the UI wasn't reachable from the network, added
   `ports: "0.0.0.0:8188:8188"` to the **packaged template**, even though the
   deployment's `.env` already had `INNEREYE_UI_PORT` for this.

The result: renders worked through the UI, but the gateway's `/v1/render` was
broken (wrong network), and the template change would have published an
unauthenticated ComfyUI on every box. The recovery was the procedure above:
remove the stray project's container, copy the fixed Dockerfile into
`~/.lobes`, set `INNEREYE_UI_PORT=0.0.0.0:8188` (the operator requires network
access and accepts that ComfyUI has no login), and recreate `comfyui` in the
`lobes` project. A 256×256 render through `/v1/render` then completed in 20 s.
The template fix shipped as PR #275 with a test pinning the compiler.
