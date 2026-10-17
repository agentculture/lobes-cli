# Rolling R1 onto a hand-kept deployment

This runbook takes an existing `~/.lobes` from a 0.86 scaffold to the R1
scaffold (**lobes-cli 0.89.0**: the 0.87/0.88 lane-safety bundle, the vLLM
0.31.0 tool-parser plugin fix and `lobes governor`) **without re-scaffolding
over hand edits**. It is for an operator or an agent working on one box of the
mesh (Spark, spark2, Thor, Orin) during the vLLM 0.31.0 move (issue #310).

The rules for touching a live box are in the
[`lobes-deploy`](../../.claude/skills/lobes-deploy/SKILL.md) skill and
[`operating-a-deployment.md`](../operating-a-deployment.md). This page doesn't
repeat them; it lists what R1 adds and where each piece goes on a box whose
compose files were copied once and then edited by hand. The contract for every
knob, file and finding is
[`contracts/lane-safety-runtime.md`](../contracts/lane-safety-runtime.md).

**Status.** The file steps (1 to 8) were dry-run once, on 2026-10-17, against a
sanitised copy of the DGX Spark's `~/.lobes` in scratch space. The record is
[`evidence/2026-10-17-dryrun-r1-rollout-spark-copy.txt`](../evidence/2026-10-17-dryrun-r1-rollout-spark-copy.txt).
The compose merge was checked by parsing the files, not by `docker compose
config`, and no lane was recreated. **No live box has run this procedure yet**
(DECLARED/UNVALIDATED, #108).

## What R1 adds, and where each piece goes

A 0.86 scaffold already runs every vLLM lane under `mg-logwrap.sh` with the
`./logs` and wrapper bind mounts and `restart: unless-stopped`. R1 adds the
following. The **Where** column applies to a hand-kept box.

| Piece | What the R1 template has | Where it goes on a hand-kept box |
|---|---|---|
| Boot hold, stagger, start counter, memory gate, `.held`/`.gated` markers | Code in `mg-logwrap.sh`; the compose is unchanged for wrapped lanes | `lobes doctor --refresh-logwrap --apply` (step 5) |
| Tool-parser plugin fix (vLLM 0.31.0 `strict_level`) | `qwen3_thinking_tool_parser.py` (works on the old nightly too) | `lobes doctor --refresh-plugin --apply` (step 6) |
| Compile cache and FlashInfer JIT cache | Per vLLM lane: `VLLM_CACHE_ROOT` and `FLASHINFER_WORKSPACE_BASE` env, plus `<lane>` and `<lane>-flashinfer` binds at `/cache/vllm` and `/cache/flashinfer` | `docker-compose.override.yml`: paste the lines `lobes doctor` prints for `compile_cache_unmounted` (step 4) |
| Cache directories on the host | Created as you by `lobes init` | `lobes doctor --fix --apply`, **after** the override (step 7) |
| Longer `healthcheck.start_period` (the hold must fit inside it) | vLLM lanes 600 to 780 s, gateway 20 to 200 s, comfyui 90 to 270 s, realtime 30 to 210 s, chatterbox 120 to 300 s (Hebrew 300 to 480 s), stt 600 to 780 s, embed sidecars +180 s | Override: `healthcheck: {start_period: …}` per service (step 4) |
| Gateway under the wrapper (boot hold) | `entrypoint` + `command`, `MG_LOG_DIR`/`MG_LOG_NAME`, `LOBES_BOOT_*` passthrough, `./logs` and wrapper binds | Override, merged into the box's existing `gateway:` block (step 4) |
| Thermal shed signal (#304, #311) | `LOBES_THERMAL_ZONES`, `LOBES_THERMAL_SHED_C`, `LOBES_THERMAL_RESUME_C` passed to the gateway | Override env (step 4); the values come from `doctor --fix` (step 7) |
| Audio and sidecar lanes under the wrapper | stt, chatterbox, realtime, `embed-gemma2-embed`: `entrypoint` + respelled image `command`, log env, binds; `llamacpp-primary` and realtime: explicit `LOBES_BOOT_*` (no `env_file`) | Override, only for services the box runs (step 4) |
| Restart policy | **Unchanged**: `restart: unless-stopped` on every service | Nothing to add. Verify it (step 3): the reboot boot hold and the start counter's exit-75 path both depend on it |
| `logs/lane-state/` | Created as you by every lifecycle verb | `lobes up <role> --apply` creates it; with raw compose, `mkdir -p ~/.lobes/logs/lane-state` first |
| Missing `.env` knobs | `env.example` | `lobes doctor --fix --apply` appends missing keys only (step 7). The wrapper has built-in defaults (`LOBES_BOOT_DELAY_S=120`, `LOBES_START_MAX=3`, `LOBES_MEM_GATE=1`), so a lane-safety knob that is absent from `.env` is still on |
| Gateway image at R1 | `MODEL_GEAR_VERSION` | `lobes doctor --repin-version --apply`, then re-image the gateway in the window (step 9) |
| Host side (optional) | The machine CLI's `grant-permissions`, `lobes watcher` | Step 8 |

### Override, not an edit of `docker-compose.yml`

Put every compose change in `docker-compose.override.yml`. The reasons:

- `lobes init` never writes the override (it is operator-owned), so the
  changes survive any later re-render of the shape file.
- `lobes doctor` already prints its `compile_cache_unmounted` remediation as
  override lines, and the doctor check reads the merged chain, so the finding
  clears once the lines are in.
- The hand-kept base file stays byte-identical, so its history (and the
  deployment lock's `[files]` digest, where a box has one) changes in one place.

Compose merge rules decide which pieces are safe in an override:

| Attribute | Merge rule | Safe in an override? |
|---|---|---|
| `environment` | Merged by variable name | Yes. A key the base already sets is replaced, so check first |
| `volumes` | Merged by **container target path** | Yes, for new targets (`/cache/vllm`, `/cache/flashinfer`, `/logs/model-gear`, `/usr/local/bin/mg-logwrap`). A base mount at the same target is replaced |
| `healthcheck` | Merged key by key; only `healthcheck.test` is replaced | Yes: setting `start_period` alone keeps the base `test`, `interval` and `retries` |
| `entrypoint`, `command` | **Replaced whole**, never appended | Yes, but only as complete values. Setting `entrypoint` also drops the image's own `CMD`, so always set `command` beside it, respelling the image's `ENTRYPOINT`/`CMD` |
| `restart`, `profiles` | Replaced | Nothing to change in R1 |

Use a reviewed edit of `docker-compose.yml` only when the override can't
express the change. Two cases: a base mount at the same container target with a
different source (the override would silently replace it; decide which is
right), or a value that must be removed (prefer the override's `!reset` tag,
which the Spark's shape file already uses).

Two YAML traps:

- **One block per service per file.** The Spark's override already has
  `gateway:`, `stt:`, `chatterbox:`, `realtime:` and `comfyui:` blocks. Merge the
  new lines into those blocks. A second `gateway:` key in the same file is a
  duplicate key, not a merge. The same goes for a second `environment:` or
  `volumes:` inside one block.
- **Respell from the box's own images.** The template's respelled `command`
  values match the packaged Dockerfiles. A box with `*.local` Dockerfiles may
  differ. Check each image before you write its `command`:
  `docker image inspect <image> --format '{{json .Config.Entrypoint}} {{json .Config.Cmd}}'`
  (read-only). An NGC-based image's `ENTRYPOINT` (`/opt/nvidia/nvidia_entrypoint.sh`)
  goes first in `command`, as in the packaged `docker-compose.audio.yml`.

When the box is re-scaffolded from an R1-or-later template one day, these
override lines become duplicates of the base. That is harmless (same targets,
same keys), but delete them then.

## Before you start

- Read the box's notes below and the `lobes-deploy` skill.
- Plan a recreate window. Steps 1 to 8 only change files on disk. Running
  containers keep their old configuration and the old wrapper inode until they
  are recreated, so you can **prepare everything now and recreate later**. On
  spark2 the recreate is combined with the engine switch in one window (#310
  plan t13).
- Between preparation and the window, a container docker restarts on its own
  (a crash, or a host reboot) keeps its old compose configuration but should
  start with the new `mg-logwrap.sh`, because a file bind mount is re-resolved
  at container start. The new wrapper works with the old configuration (no
  compile-cache env means no cache, as before). This was reasoned from Docker's
  mount behaviour, not tested.

## The procedure

Run every command on the box, as the login user, from a shell outside any
lobes-cli checkout. `D=~/.lobes` below; use the box's `$LOBES_DIR` if it has
one.

1. **Install R1 with uv.** On Thor, Orin and spark2 there is no lobes CLI yet;
   on the Spark, upgrade the existing uv tool (0.83.0 on 2026-10-17):

   ```bash
   uv tool install 'lobes-cli==0.89.0'
   lobes --version            # want: lobes 0.89.0
   ```

   If uv answers that the tool is already installed at another version, add
   `--reinstall`.

   Install the box's machine CLI as its own uv tool, so `spark`, `thor` or
   `orin` is on `PATH` (lobes looks the CLI up by name):
   `uv tool install 'dgx-spark-cli>=0.9.1'`, `'jetson-thor-cli>=0.6.1'` or
   `'jetson-orin-cli>=0.7.1'`. These are the first releases with the
   `window`, `drop-caches` and `grant-permissions` verbs; an older machine CLI
   (the Spark had dgx-spark-cli 0.7.0 on 2026-10-17) must be upgraded the same
   way.

2. **Back up, dated.** These copies are the rollback:

   ```bash
   TS=$(date -u +%Y%m%dT%H%M%SZ)-r1
   cd $D && for f in .env docker-compose*.yml mg-logwrap.sh qwen3_thinking_tool_parser.py; do
     [ -e "$f" ] && cp -p "$f" "$f.bak-$TS"; done
   ls -a | grep "bak-$TS"
   ```

   Keep the `.bak-<stamp>` suffix after `.yml`. A name that still ends in
   `.yml` (such as `override.bak.yml`) is read by `lobes doctor`'s compose scan
   as part of the deployment. Doctor's own refresh lanes write a plain `.bak`,
   which a second run overwrites; the dated copies are the ones to restore.

3. **Read the box's state.**

   ```bash
   lobes fleet files --json       # the -f chain; docker-compose.override.yml must be in it
   lobes doctor > /tmp/doctor-before-$TS.txt; grep '^\[' /tmp/doctor-before-$TS.txt
   docker inspect model-gear-gateway \
     --format '{{index .Config.Labels "com.docker.compose.project.config_files"}}'
   ```

   The chain the CLI prints should match the gateway's label. Then compare the
   packaged template with the box's compose for the lane-safety blocks:

   ```bash
   T=$(ls -d "$(uv tool dir)"/lobes-cli/lib/python3.*/site-packages/lobes/templates)
   grep -nE 'cache/vllm|cache/flashinfer|LOBES_BOOT|LOBES_THERMAL|mg-logwrap|start_period|restart:' \
     $T/fleet/docker-compose.yml > /tmp/tpl-$TS.txt
   grep -nE 'cache/vllm|cache/flashinfer|LOBES_BOOT|LOBES_THERMAL|mg-logwrap|start_period|restart:' \
     $D/docker-compose*.yml > /tmp/box-$TS.txt
   ```

   For each running service, note what the table above says it lacks. Check
   that every service has `restart: unless-stopped`; R1 adds none.

4. **Write the override.** Back up first (step 2), then add the lines to
   `$D/docker-compose.override.yml`, merging into existing service blocks:

   - The compile-cache lines `lobes doctor` printed for
     `compile_cache_unmounted`, verbatim, one block per vLLM service it named.
     Add `healthcheck: {start_period: 780s}` to each.
   - The gateway: `MG_LOG_DIR=/logs/model-gear`, `MG_LOG_NAME=gateway`, the
     three `LOBES_BOOT_*` passthroughs and the three `LOBES_THERMAL_*`
     passthroughs in `environment`; the `./logs` and `./mg-logwrap.sh` binds;
     `entrypoint: ["bash", "/usr/local/bin/mg-logwrap"]`;
     `command: ["python", "-m", "lobes.gateway"]`; `start_period: 200s`. Copy
     the exact lines from the packaged template's `gateway:` service.
   - Each audio or sidecar service the box runs, copied the same way from the
     packaged `docker-compose.audio.yml`, `docker-compose.audio-he.yml` or
     `docker-compose.embed.yml`, with `command` respelled from the box's image
     (see the YAML traps above).
   - The `start_period` raises for any other wrapped service (for example
     `comfyui: 270s`).

   Then check that the YAML still parses and that doctor sees the mounts:
   `lobes doctor | grep compile_cache_unmounted` should say every vLLM service
   mounts both caches. Then validate the full merge with compose itself
   (read-only). Where the repo is checked out, use the skill's wrapper,
   `lobes-compose.sh config --quiet`. Elsewhere, from the deployment dir,
   `cd $D && docker compose $(lobes fleet files) config --quiet`. This check
   was not run in the dry run.

5. **Refresh the wrapper.** `lobes doctor --refresh-logwrap` prints the plan;
   `lobes doctor --refresh-logwrap --apply` writes `mg-logwrap.sh.bak`, then the
   packaged wrapper (atomic rename, so running containers keep the old inode).

6. **Refresh the plugin.** `lobes doctor --refresh-plugin`, then
   `lobes doctor --refresh-plugin --apply`. It writes
   `qwen3_thinking_tool_parser.py.bak`, then the packaged copy. Only the cortex
   lane mounts it.

7. **Heal `.env` and create the cache directories.** `lobes doctor --fix`
   prints the plan; review every `would append` line before you apply it,
   because `--fix` appends **every** missing required key, not only the
   lane-safety ones. Then `lobes doctor --fix --apply`. It never rewrites an
   existing line. It must run **after** step 4: it creates one directory per
   `<lane>` and `<lane>-flashinfer` bind it finds in the compose files.

   Two consequences to know:

   - On a Spark, `--fix` appends `LOBES_THERMAL_ZONES`, `LOBES_THERMAL_SHED_C`
     and `LOBES_THERMAL_RESUME_C` from the card profile. With the gateway
     passthrough from step 4, the recreated gateway **starts shedding
     full-tier requests (429) above the shed band**. Those Spark bands come from
     a batch-job governor, not a serving measurement. To keep the signal off,
     set `LOBES_THERMAL_SHED_C=` (empty) after the append.
   - Other appended keys can change behaviour where the compose reads them
     with a different default. Compare each appended key's value with the
     `${KEY:-default}` the box's compose uses for it.

8. **Host side (optional, once per box).** Run the machine CLI's
   `grant-permissions` (needs sudo once) so lifecycle verbs and the watcher
   can clear the page cache: `lobes doctor` stops reporting
   `drop_caches_grant_missing`. Then, if the operator wants it,
   `lobes watcher install --apply` (a user unit; needs linger, or `--system`).

9. **Recreate, in the window.** New mounts, entrypoints and environment take
   effect only when a container is **recreated**; a `docker restart` keeps the
   old configuration. Recreate one service at a time and wait for it to be
   healthy before the next:

   ```bash
   lobes up <role> --apply     # cortex, reranker, embedder, hand, worker, …
   ```

   `lobes up` clears the page cache (when granted), resets the lane's start
   counter, and creates `logs/lane-state/` and the cache directories as you
   before the container starts. For a service `lobes up` doesn't name, use the
   skill's wrapper, `--apply` first:
   `lobes-compose.sh --apply up -d --no-deps <service>`, or on a box without
   the repo, `cd $D && mkdir -p logs/lane-state && docker compose $(lobes fleet
   files) up -d --no-deps <service>`. If a container shows a creation time
   older than the override edit, add `--force-recreate`.

   The gateway: `lobes doctor --repin-version` then `--repin-version --apply`
   sets `MODEL_GEAR_VERSION=0.89.0`. If `.env` sets
   `GATEWAY_PIP_EXTRA_INDEX_URL` to TestPyPI, comment it out first, or the
   build finds no final release. Then `lobes up gateway --build --apply`
   (0.89.0 carries the audio overlay into the gateway's chain when one is
   scaffolded). Recreate the gateway last, because it fronts every lane.

## Verify

After each recreate, and again at the end:

```bash
lobes doctor | grep '^\['
#   want: [ok] logwrap_stale, [ok] plugin_stale, [ok] compile_cache_unmounted,
#         [ok] gateway_version_match; lane_state_writable ok or absent
docker inspect <container> --format '{{.Created}} {{json .Config.Entrypoint}}'
docker inspect <container> --format '{{range .Mounts}}{{.Destination}} {{end}}'
#   want: /cache/vllm and /cache/flashinfer on every vLLM lane; /logs/model-gear everywhere
docker exec <container> sh -c 'echo $VLLM_CACHE_ROOT $FLASHINFER_WORKSPACE_BASE'
ls ~/.lobes/logs/lane-state/          # a <lane>.starts line per start
ls ~/.lobes/compile-cache/<lane>/     # torch_compile_cache appears after the first boot
grep -h 'lobes: host up' ~/.lobes/logs/*.log | tail   # only after a reboot
curl -s localhost:${VLLM_PORT:-8000}/health           # version 0.89.0
docker exec model-gear-gateway sh -c 'echo $AUDIO_URL'  # unchanged on an audio box
```

Then one real request per recreated role through the gateway, as the
`lobes-deploy` skill describes. On a mesh member, check that the other members
see this box again (`GET /mesh/roster`) before you recreate the next box.

## Rollback

Restore the dated backups and recreate the lanes you changed:

```bash
cd ~/.lobes && TS=<the stamp from step 2>
for f in .env docker-compose*.yml mg-logwrap.sh qwen3_thinking_tool_parser.py; do
  [ -e "$f.bak-$TS" ] && cp -p "$f.bak-$TS" "$f"; done
# Files R1 added (remove only if they did not exist before):
rm -f cf-tunnel.env.example mg-logwrap.sh.bak qwen3_thinking_tool_parser.py.bak
```

Then recreate each changed service again (step 9). Put the old gateway pin
back the same way; it lives in the restored `.env`. The `compile-cache/`
directory can stay: nothing reads it once the mounts are gone. Delete it with
no lane running if you want the space back. `lobes watcher uninstall --apply`
removes the watcher. To roll the CLI back,
`uv tool install --force 'lobes-cli==<previous>'` (on boxes that had none,
`uv tool uninstall lobes-cli`).

In the dry run, restoring the backups and removing the added files gave a tree
identical to the original copy (`diff -r` clean).

## Box notes

- **DGX Spark.** The compose files are hand-kept. Never run `lobes init
  --force` or re-scaffold. Every compose action goes through the `lobes-deploy`
  skill's `scripts/lobes-compose.sh` or a `lobes up` verb. The Spark's override
  already carries the Hebrew audio deltas and the gateway passthroughs; merge
  into those blocks. Its dgx-spark-cli was 0.7.0 on 2026-10-17, older than the
  0.9.1 lane safety needs (step 1). Run heavy work under the thermal governor
  (`lobes governor`).
- **spark2.** The login user is `spark2`, and sudo needs a password, so step 8
  needs the operator. uv is not installed: install it with the official
  installer, `curl -LsSf https://astral.sh/uv/install.sh | sh` (it installs to
  `~/.local/bin`; open a new shell or `source ~/.local/bin/env`), then step 1.
  Prepare steps 1 to 8 ahead; the recreate in step 9 is the **same window** as
  the switch to vLLM 0.31.0 (plan t13), and it waits until the Spark's cortex is
  verified, so plain `cortex` stays answerable mesh-wide.
- **Jetson AGX Thor (measured on the first live run, 2026-10-17).** Three
  things to know before step 9:
  - **Clear the page cache after the old container stops, not only before.**
    `lobes up` clears it first and then recreates, but a stopped engine's
    memory stays "used" until a clear runs after it stops. A large lane
    (the worker, about 58 GB) then waits at the memory gate
    (`logs/lane-state/<lane>.gated`) indefinitely. Run
    `sync; sudo sysctl -w vm.drop_caches=3` once the old container is gone,
    or install the host watcher (step 8). On 2026-10-17 that clear returned
    74 GiB.
  - **The worker keeps its own warm caches** at `/root/.cache/{vllm,flashinfer}`.
    Don't move them to `/cache/*`: cold caches crash most Thor boots (#309).
  - **The gateway can't write `./logs`.** It runs as uid 10001, so the wrapper
    skips its log file and start counter (by design; the boot hold still runs).
- **Jetson AGX Thor (general).** Its `~/.bashrc` exports `GATEWAY_API_KEY`, and a compose
  run from that shell would arm the gateway's inbound key gate. Prefix every
  compose and `lobes up` call with `env -u GATEWAY_API_KEY`. The Thor is shared
  with another agent's work: agree the window before you recreate anything. It
  has a variation-catalog entry (`deployments/jetson-agx-thor__thor-worker/`).
  If its deployment dir holds a `deployment.lock.toml`, `lobes doctor` reports
  `lock_drift` after this procedure until the lock is re-captured (a later
  #310 task).
- **Jetson AGX Orin.** Zero swap. Its override already has the `vllm-embed`
  compile-cache mount from the 2026-10-17 hardware pass; merge, don't
  duplicate. Its chain includes `docker-compose.gpu.yml` (`runtime: nvidia`).
  After heavy I/O (a download, a copy), evict the page cache before booting a
  lane, because CUDA free memory excludes it and vLLM refuses to boot. Cap any
  side process with `--memory`. Its catalog entry
  (`deployments/jetson-agx-orin__orin-embed/`) goes stale the same way as the
  Thor's.
