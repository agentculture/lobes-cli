# Validate Tiers — Operator Runbook

Three-tier generate-fleet validation for issue #68, task t9.
Runs `scripts/validate-tiers.sh` against an already-running fleet to confirm
that tier alias routing, pressure downgrade, and manual override all behave
correctly end-to-end on the live hardware.

## Prerequisites

Before running the script:

1. **Fleet up with all five gears.**
   The fleet must be running with `COMPOSE_PROFILES` including `middle` and `minor`:

   ```bash
   COMPOSE_PROFILES=minor,middle lobes fleet up --apply
   # or set COMPOSE_PROFILES=minor,middle in your deployment .env and then:
   lobes fleet up --apply
   ```

   Confirm all five gears (primary, middle, minor, embed, rerank) and the
   gateway are healthy before proceeding:

   ```bash
   lobes fleet status --compose-dir ~/.model-gear
   ```

2. **Gateway on the expected port.**
   Default: `http://localhost:8001/v1`. On the DGX Spark the deployment dir is
   `~/.model-gear` and `VLLM_PORT` in `.env` sets the host port. Pass
   `--base-url` if your port differs.

3. **Required tools:** `bash`, `curl`, `jq`, `nvidia-smi`, `docker` (compose v2),
   `lobes` CLI.

## How to Run

```bash
# Minimum invocation (defaults: localhost:8001, ~/.model-gear or $LOBES_DIR):
./scripts/validate-tiers.sh

# Full explicit form:
./scripts/validate-tiers.sh \
  --compose-dir ~/.model-gear \
  --base-url   http://localhost:8001/v1 \
  --out        /tmp/validate-tiers-evidence.txt
```

Exit code `0` means all hard checks (A, C, D, E) passed. Exit code `1` means
at least one hard check failed. Checks B and F are non-fatal evidence checks.

## What Each Check Proves

### A — Fleet health

Confirms all five gears are in `running*` state and the gateway responds to
`/health`. The base gears (primary, embed, rerank) are queried via
`lobes fleet status --json`; the opt-in profile containers (minor, middle) are
checked directly via `docker inspect` because `fleet_containers()` only covers
the always-warm base set.

**Maps to t9 acceptance criterion:** all five co-resident gears fit on the GB10
with the documented budget (0.79) and start successfully.

### B — Memory budget (informational)

Runs `nvidia-smi` and prints live GPU memory. Compare against the documented
budget:

| Gear | `--gpu-memory-utilization` | Approx GiB |
|---|---|---|
| primary (hard, 27B) | 0.45 | ~56 |
| middle (normal, 14B) | 0.12 | ~15 |
| minor (cheap, 4B) | 0.10 | ~13 |
| embed (0.6B) | 0.06 | ~7 |
| rerank (0.6B) | 0.06 | ~7 |
| **Total** | **0.79** | **~98 / 128 GB** |

Non-fatal: nvidia-smi parse quirks do not block the exit code.

### C — Tier alias routing

POSTs `model=normal` to `POST /v1/chat/completions` and asserts **both**:

- `X-Lobes-Tier: normal` response header
- `body.model` contains `Qwen3-14B`

Smoke-tests `model=cheap` (expect `X-Lobes-Tier: cheap`) and `model=hard`
(expect `X-Lobes-Tier: hard`).

**Maps to t9 acceptance criterion:** callers address the generate lane by
capability-tier alias and the gateway routes to the correct gear.

### D — Pressure downgrade (simulated)

This check simulates an overloaded host **without actually stressing memory**:

1. Read the live `swap_used_percent` from `lobes status --pressure --json`.
2. Lower `LOBES_SWAP_DEGRADED_THRESHOLD` to `floor(swap%) - 1` by writing a
   temporary `docker-compose.validate-tiers.override.yml` in the deployment dir:

   ```yaml
   services:
     gateway:
       environment:
         - LOBES_SWAP_DEGRADED_THRESHOLD=<lowered>
   ```

3. Force-recreate **only the gateway** (`--no-deps --force-recreate gateway`).
   The recreate mirrors how `lobes` itself composes the fleet — base
   `docker-compose.yml` **plus the audio overlay when scaffolded** — with the
   override merged on top via explicit `-f`, so the gateway keeps its full
   config (including `/v1/audio/*` routing) and only the threshold changes.
   The vLLM backends are untouched. The gateway re-imports
   `lobes.gateway._pressure_policy` at container start, picking up the lowered
   threshold.
4. POST `model=hard` and assert `X-Lobes-Tier: cheap` and
   `X-Lobes-Tier-Reason: pressure`.

The `LOBES_SWAP_DEGRADED_THRESHOLD` constant is read at **module import time**
inside the gateway container. Changing it requires a gateway restart; the
vLLM backends do not need to restart.

**Why the trap matters:** the override file must be removed and the
gateway must be recreated at defaults before the script exits, or the fleet will
stay with a degraded threshold that blocks all `hard` requests permanently.
The `trap '_cleanup' EXIT` in the script handles this unconditionally — even if
the script is killed with Ctrl-C or fails partway through D.

**Maps to t9 acceptance criterion:** swap or iowait pressure above the
degraded threshold downgrades `hard` requests to `cheap`.

### E — Manual override

With the same lowered threshold still active from D, POSTs `model=hard` **with
`X-Lobes-Override: 1`** and asserts:

- `X-Lobes-Tier: hard` (override bypassed the downgrade)
- `X-Lobes-Tier-Reason: manual_override`

The gateway's truthy tokens for `X-Lobes-Override` are `1`, `true`, `yes`.

**Maps to t9 acceptance criterion:** operators can force the requested tier
regardless of pressure when they explicitly set `X-Lobes-Override`.

### F — Pressure delta evidence (informational)

Fires three completions against `cheap` + `normal` (4B/14B, with override to
bypass the still-lowered threshold) and three against `hard` (27B, with
override). `lobes status --pressure --json` is sampled before and after each
burst to capture `swap_used_percent` and `iowait_percent` deltas. The evidence
is printed to stdout and optionally written to `--out`.

The expected pattern: the cheap/normal burst produces smaller or equal
swap/iowait delta compared with the hard burst, reflecting that the 27B primary
is more memory-bandwidth-intensive than the smaller gears.

Non-fatal: a reverse delta is not an automatic failure — transient OS activity
can dominate on a shared box. Treat this as evidence, not a gate.

**Maps to t9 acceptance criterion:** documents that tiered routing keeps host
pressure lower for routine workloads.

## Reading the Pressure Delta Evidence File

When `--out FILE` is passed, the file contains timestamped pressure readings
for both bursts. Example:

```text
F: Pressure delta evidence  (2026-06-30T10:15:03Z)
compose-dir: /home/user/.model-gear

cheap+normal burst (4B/14B gears, 3 completions with X-Lobes-Override):
  swap_before:   22.4%
  swap_after:    22.6%
  iowait_before: 0.8%
  iowait_after:  0.9%

hard burst (27B gear, 3 completions with X-Lobes-Override):
  swap_before:   22.6%
  swap_after:    24.1%
  iowait_before: 0.9%
  iowait_after:  1.2%
```

Larger deltas on the hard burst confirm the 27B primary uses more memory
bandwidth — the motivation for routing routine tasks to cheaper tiers.

## Troubleshooting

**Check A fails for minor/middle:** confirm `COMPOSE_PROFILES=minor,middle` is
set in the deployment `.env` and the fleet was started **after** that change.
Profiles are read at `docker compose up` time.

**Check D fails (downgrade not firing):** the gateway container's
`LOBES_SWAP_DEGRADED_THRESHOLD` env var was not set. Confirm the
`docker-compose.validate-tiers.override.yml` was written to `$COMPOSE_DIR`
before the `--force-recreate` call and that `docker compose up` picked it up
(it is passed explicitly via `-f`). Check
`docker inspect model-gear-gateway | grep -A5 Env`.

**Check C body.model mismatch:** the middle gear may not have finished loading
(vLLM load takes several minutes). Wait until `lobes fleet status` shows
`running (healthy)` for `model-gear-vllm-middle`.

**Gateway not restored after failure:** if the script crashes before the trap
can run, delete the override file and recreate the gateway. Include the audio
overlay if your deployment uses it, matching how `lobes` composes the fleet:

```bash
rm ~/.model-gear/docker-compose.validate-tiers.override.yml
docker compose --project-directory ~/.model-gear \
  -f ~/.model-gear/docker-compose.yml \
  -f ~/.model-gear/docker-compose.audio.yml \
  up -d --no-deps --force-recreate gateway
# Omit the audio -f line if you did not init with --audio. Simplest of all:
# `lobes fleet up --apply` recreates the fleet with the correct compose files.
```
