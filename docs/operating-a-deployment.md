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

## Lane safety: what a restart does now

Issues #304, #307, #308 and #309 added five protections around lane start,
stop and crash. This section is for operators and for agents driving a box.
The contract (every knob, file and format) is
[`docs/contracts/lane-safety-runtime.md`](contracts/lane-safety-runtime.md);
the machine-CLI half is
[`docs/contracts/machine-cli-window.md`](contracts/machine-cli-window.md). Every
knob lives in the deployment's `.env`, ships on, and has an **off value** that
restores the earlier behaviour (table at the end of this section). The Thor is
code-only during this delivery, so every Thor line here is
**DECLARED/UNVALIDATED (#108)**.

**Boot hold (#308).** After a host reboot, dockerd restarts containers by
restart policy and ignores compose `depends_on` (assumption c32, unverified on
DGX OS and JetPack). Before, every lane raced for unified memory at once, which
is the first-boot failure `machine-profiles.md` records for the Thor. Now each
container's own `mg-logwrap.sh` holds until host uptime reaches
`LOBES_BOOT_DELAY_S` (default 120), plus a stagger of `LOBES_BOOT_STAGGER_S`
(default 20) times the lane's index in `LOBES_BOOT_ORDER`. The order comes from
the card's `[host_env]`: Orin `multimodal,embed,rerank` (measured 2026-08-04),
Thor `embed,rerank` (DECLARED/UNVALIDATED), Spark none (no order measured). The
hold is logged once in the lane log as `lobes: host up ...`. A hand-kept box
with an old `mg-logwrap.sh` needs `lobes doctor --refresh-logwrap --apply`
(the `logwrap_stale` finding); a compose file that lacks the knobs needs
`docker-compose.override.yml` lines, never `init --force`.

**Compile cache (#307).** vLLM's compile cache used to die with the container,
so every recreate re-compiled. Each vLLM lane now binds
`${LOBES_COMPILE_CACHE_DIR:-./compile-cache}/<lane>` at `/cache/vllm` and sets
`VLLM_CACHE_ROOT`. The cache is content-hashed, so a vLLM upgrade (#310)
leaves stale entries unread rather than wrong. This was verified by read-only
inspection on both the pinned nightly and v0.31.0 (contract section 6). To
wipe a lane, delete its directory with the lane stopped; no sudo is needed
because `lobes init` and `lobes doctor --fix --apply` create the directories as
you. `lobes doctor` reports `compile_cache_unmounted` and `compile_cache_size`.
FlashInfer keeps its JIT cache under the container's home, so each lane also
binds `<lane>-flashinfer` at `/cache/flashinfer` and sets
`FLASHINFER_WORKSPACE_BASE` from `LOBES_FLASHINFER_WORKSPACE_BASE`; a fresh
container then skips the JIT build where the Thor cold-boot crash happens
(vllm#62149, deviation d7).

**Crash backoff and the memory gate (#308).** Before, a lane that could not get
memory crash-looped and each crash made the next boot harder. Now the wrapper
appends a line to `<deployment>/logs/lane-state/<lane>.starts` on every start;
the 4th start inside `LOBES_START_WINDOW_S` (600 s) holds and logs how to clear
it (`lobes up <role> --apply`, or delete the file). Before a vLLM lane starts,
the memory gate waits (polling every 5 s, up to `LOBES_MEM_GATE_TIMEOUT_S`) for
`ceil(util x MemTotal)` of `MemAvailable`, and keeps a
`lane-state/<lane>.gated` marker while waiting. On timeout it exits 75, which
counts as a start. Known limit: on Jetson, freed memory can stay out of
`MemAvailable` until the page cache is dropped, so the gate can wait when vLLM
would fit; the start counter is the backstop.

**Cache clear, grant and watcher (#309).** `lobes up`, `serve`, `switch`,
`stop` and `fleet` clear the page cache around down/up (on a detected Spark,
Thor or Orin; elsewhere the step is skipped silently) and delete the
start-counter files of the lanes they start on purpose, so deliberate restarts
never count. The clear needs a passwordless rule. `lobes doctor` reports
`drop_caches_grant_missing`; the machine CLI's `grant-permissions` installs the
single rule (the Thor already has an equivalent one, which is reused). With no
grant, the verb prints one stderr line and carries on; it never prompts and
never fails. For a lane stuck at the gate, `lobes watcher install --apply`
(user unit; needs linger, or `--system`) polls the `.gated` markers and runs
`drop-caches` at most once per `LOBES_WATCHER_MIN_INTERVAL_S` (300). The watcher
never runs a docker command. Check it with `lobes watcher status`; `lobes
doctor` reports `watcher_inactive`. The machine-CLI verbs (`window`,
`drop-caches`, `grant-permissions`) ship in dgx-spark-cli 0.9.1, jetson-thor-cli
0.6.1 and jetson-orin-cli 0.7.1 (install with `lobes-cli[spark|thor|orin]`).
Without the machine CLI, lobes falls back to `sudo -n` for the clear only.

**Thermal shedding (#304).** The gateway can shed full-tier requests with a 429
when the card runs hot; see `gateway-fleet.md`. A deployment scaffolded before
issue #311 lacks the gateway passthrough and stays inert until its compose is
updated, and the Spark bands are not a serving measurement.

**Benchmark window (#304).** `lobes benchmark` locks clocks for its run and
records the host conditions; see `measuring-lane-performance.md`.

**What none of this does:** no lobes code pauses, `docker update`s or restarts a
serving container, nothing clears the page cache inside a container, and no
clock or fan changes outside a benchmark window. The live Spark acceptance
under sustained load has **not** run (issue #313).

| Feature | Off value (in `.env`) |
|---|---|
| Boot hold and stagger | `LOBES_BOOT_DELAY_S=0` (stagger alone: `LOBES_BOOT_STAGGER_S=0`) |
| Start counter | `LOBES_START_MAX=0` |
| Memory gate | `LOBES_MEM_GATE=0` |
| Compile cache | `LOBES_VLLM_CACHE_ROOT=/root/.cache/vllm` |
| FlashInfer JIT cache | `LOBES_FLASHINFER_WORKSPACE_BASE=/root` |
| Thermal shed | `LOBES_THERMAL_SHED_C=` (empty) |
| Watcher | `lobes watcher uninstall --apply` |

Each off value is proved by the per-feature tests named in
`tests/test_lane_safety_knobs_documented.py`.

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
