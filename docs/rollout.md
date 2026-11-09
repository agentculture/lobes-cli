# Gradual fleet rollout

> **DECLARED/UNVALIDATED (#108).** Everything below describes what shipped in
> the code and is exercised by unit tests with injected transports. No
> rollout, residents file, export or site has been run against the live mesh;
> that is the live acceptance (plan task t20), and its transcript will land
> under `docs/evidence/`. Until then read each behaviour as "implemented and
> unit-tested", not "measured".

`lobes rollout` upgrades the `lobes-cli` release across the mesh's resident
boxes one wave at a time, verifies each member through the mesh, and stops on
the first regression. It replaces a hand-run upgrade, and it deliberately does
less than a deploy tool: see [What it never touches](#what-it-never-touches).

## The before-state (2026-11-09)

The spec (`docs/specs/2026-11-09-gradual-fleet-rollout.md`) records how the
upgrade to 0.94.0 went by hand. No transcript for it exists under
`docs/evidence/`; these facts are cited as recorded in the spec.

- **Four ad-hoc scripts.** Per box: SSH in, `uv tool install`,
  `lobes doctor --repin-version`, `lobes up gateway --build`, then a hand test.
  Versions 0.89.0, 0.89.2, a dev build and 0.93.0 coexisted.
- **An unpublished release.** 0.94.0 merged on 2026-11-04 but never reached
  PyPI (`publish.yml` run 41882958341: the test job ran out of disk and the
  publish step was skipped); nobody noticed for five days.
- **PyPI CDN lag.** After the rerun published it, `pip` inside
  `python:3.12-slim` still could not resolve `lobes-cli==0.94.0` for several
  minutes, so the first gateway build failed.
- **A doctor exit code.** `lobes doctor --repin-version --apply` exits
  non-zero while `gateway_version_match` fails, which is the very skew it is
  about to fix. A `set -e` script stopped there on spark2.
- **Missing `stt-en` gateway keys.** spark2's hand-kept compose predated the
  lane and lacked `STT_EN_BASE_URL` / `STT_EN_FEASIBLE`. After the upgrade
  `stt-en` was silently infeasible (HTTP 404) until four lines went into
  `docker-compose.override.yml`.

Each has a counterpart below: the journal and `plan` replace the scripts,
preflight catches the unpublished release and the lag, the repin step judges
the resulting `.env` rather than doctor's exit code, and drift detection
catches the missing keys.

## What it never touches

The rollout touches only:

- the lobes CLI install on the member (`uv tool install lobes-cli==<v>`);
- the `MODEL_GEAR_VERSION` pin (`lobes doctor --repin-version --apply`);
- a timestamped backup of the member's `.env`;
- the gateway image: a `lobes-gateway:<old>` tag of the running image, then
  `lobes up gateway --build --apply`;
- the per-member rollout lock (`~/.lobes/.rollout.lock`);
- the local journal (`rollout-journal.jsonl`) and `residents.toml`;
- with `--patch-drift` only, the member's `docker-compose.override.yml`.

It never writes vLLM lane images, checkpoints, shape or profile files, and
never runs `lobes init`, `switch`, or `up` for any lane but the gateway.
`tests/test_rollout_boundary.py` enforces this statically over
`lobes/rollout/*.py`, `lobes/cli/_commands/rollout.py` and
`lobes/cli/_commands/lobe.py`. It proves the code does not name such an
operation; it is not a sandbox.

## Residents

`residents.toml` lives in the deployment dir (default `~/.lobes`) and says
which members the operator can reach. Each `[resident.<name>]` (managed) or
`[guest.<name>]` (seen, not managed) has five fields: `ssh`, `user`, `port`,
`lobes_path` (absolute path of the `lobes` binary on the member; a
non-login SSH shell has no `~/.local/bin` on `PATH`) and `gateway_key` (a
boolean, never the key). A field is either detected or marked
`# fix: <field> unknown` and listed in `needs_you`. Nothing is guessed, and a
mesh name is never turned into an SSH target. Guests are never upgraded.

```text
lobes mesh resident add <name> [--apply] [--compose-dir D] [--port P] [--json]
```

Dry-run by default. The member must be on the local gateway's mesh roster and
verified; it exits 1 if it is not, or is already a resident, and 2 if the
roster is unavailable. Only the gateway port in the roster origin is detected;
fill the other fields by hand in `residents.toml`. An operator-set value is
never overwritten.

`lobes doctor` adds three findings when `residents.toml` exists (none at all
when it does not):

- `new_mesh_member`: a verified mesh member is in neither table. Skipped when
  the gateway is unreachable or `LOBES_MESH_KEY` is unset.
- `resident_field_missing`: a resident still has a `needs_you` field.
- `gateway_open`: a resident has no gateway key, so its gateway is open.

## plan

```text
lobes rollout plan <version> [--compose-dir D] [--port P] [--local NAME]
                   [--canary M] [--wave-size N] [--allow-downgrade]
                   [--template FILE] [--json]
```

Read-only; exits 0. For each resident it asks that member's own gateway for
`/health` and `/capabilities` and reads its compose files over SSH. It prints
waves (the canary first, this box last), skips (guests, already current,
offline, refused downgrades unless `--allow-downgrade`), drift, members with a
`deployment.lock.toml` (the repin will make `lock_drift` fire; re-capturing the
lock stays the operator's call) and each member's addressing.

The canary defaults to the member with the fewest roles hosted nowhere else.
`--wave-size` sets members per wave (default 1).

**Drift** is a gateway env key the target template passes that a member's
deployed compose chain does not. It is checked against this CLI's own fleet
template unless `--template` points at the target release's
`docker-compose.yml`. That is the `stt-en` gap above.

## apply

```text
lobes rollout apply <version> [--apply] [--patch-drift] [--allow-window]
                    [plan flags...]
```

Without `--apply` it is a dry run: the plan, what would be refused, and the
steps. With `--apply`:

1. **Refuse on drift** unless `--patch-drift`, which backs up and rewrites
   only the member's `docker-compose.override.yml`.
2. **Preflight** the release: `pip download --no-deps lobes-cli==<v>` inside
   the gateway's own base image (`docker run`), retried with exponential
   backoff for up to five minutes to ride out index lag. If it never
   resolves, no member is contacted.
3. **Per member, one at a time**, taking the member's lock: window check,
   back up `.env`, install the CLI, repin `MODEL_GEAR_VERSION`, tag the running
   gateway image, rebuild the gateway, wait for `/health` to report the
   target. Each step is idempotent. The window check skips a member during an
   active benchmark window unless `--allow-window`.
4. **Verify through that member's gateway**: `/health` reports the target, no
   role regressed in the `/capabilities` diff, and one real request per
   feasible and ready role succeeds. Only then does the next member start.

It never rolls back by itself. Exit codes:

- **0**: `complete` or `incomplete` (some members skipped, for example
  offline; the output says which are pending).
- **1**: drift refused (nothing was touched), or another user error.
- **2**: preflight failed, or the rollout `halted`. A halt names the member,
  the failed check, the role and the exact rollback command.

## Resume, status, rollback

Every step goes to `rollout-journal.jsonl` in the deployment dir
(secret-scrubbed). Re-running `apply <version> --apply` resumes: a verified
member is not redone, and a half-done member continues idempotently.

```text
lobes rollout status [--target V] [--compose-dir D] [--json]
```

Read-only; exits 0. Per-member state from the journal for the latest (or
given) target, plus the resume point.

```text
lobes rollout rollback --member M [--apply] [--backup PATH]
                       [--old-version V] [--port P] [--json]
```

Dry-run lists the actions; `--apply` restores the member's `.env` from the
pre-apply backup and the previous gateway image, then reports what `/health`
now says. Backup and old version come from the journal unless given. Exit 1
when they cannot be found or the member is not a usable resident; 2 when the
rollback itself fails.

## lobes lobe export

```text
lobes lobe export <role>... --member M [--base-dir D] [--out DIR]
                  [--local] [--apply] [--json]
```

Describes a role or lane's build recipe from a member: its Dockerfile, build
inputs and compose service, or, for a container started outside compose, the
image and a reconstructed `docker run`. Read-only toward the member (SSH
`cat`, `docker compose config`, `docker inspect`); `.env` and `.secrets.env`
are never read and every text is redacted. Dry-run lists the files;
`--apply` writes the bundle under `--out` (default `./lobe-export`). An
image-only lobe is described, not rebuilt.

## lobes site

```text
lobes site [--apply] [--host 127.0.0.1] [--port 9490] [--gateway URL]
           [--compose-dir D] [--json]
```

Dry-run prints the URL and routes. `--apply` serves, in the foreground, a
local page (Mesh, Rollout, Residents) and a read-only JSON API on one origin:
`/api/mesh`, `/api/rollout`, `/api/residents` and `/api/lobe-export?role=&member=`.
Only GET and HEAD are served, nothing is written, and bodies are scrubbed of
secrets. The server has no authentication; binding anything but loopback
exposes it to that network and prints a warning.

## Known limits

- `plan` does not run the preflight (it would pull an image); only
  `apply --apply` does.
- Drift is checked against this CLI's own template unless `--template` is
  given.
- `residents.toml` has no deploy-dir field: members are assumed to deploy in
  `$HOME/.lobes`.
- A rolled-back member still shows `done` in `status`: the journal never
  clears `done`.
- The site shows a split brain only when members advertise `spans` or
  `world_size` in `/capabilities`, peer memory only when a source measured it
  (the local box's `/proc/meminfo` by default), and a remote hub only when a
  tunnel probe evidences it. No gateway advertises these today, so expect a
  single-model, local-memory view.
- An image-only lobe can be described by `lobe export`, not rebuilt.
- Wire compatibility between consecutive releases is assumed, not checked:
  the mesh has no protocol-version check, so features new in the target are
  reachable only from already-upgraded fronts during the rollout window.
- Single-box deployments with no mesh roster are not supported; they keep
  `lobes doctor --repin-version` and `lobes up gateway --build`.
