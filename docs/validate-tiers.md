# Validate Tiers — Operator Runbook

Generate-fleet tier validation for issue #68, task t9 (updated for issue #69
vocabulary: `main`/`minor`/`multimodal` with back-compat `hard`/`cheap`/`normal`).
Runs `scripts/validate-tiers.sh` against an already-running fleet to confirm
that tier alias routing, pressure busy-backpressure, and manual override all
behave correctly end-to-end on the live hardware.

## Prerequisites

Before running the script:

1. **Fleet up with all required gears.**
   The default fleet starts four gears automatically (primary, multimodal, embed,
   rerank). For full tier coverage, also activate the opt-in `minor` (4B):

   ```bash
   COMPOSE_PROFILES=minor lobes fleet up --apply
   # or set COMPOSE_PROFILES=minor in your deployment .env and then:
   lobes fleet up --apply
   ```

   Confirm all five gears (primary, multimodal, minor, embed, rerank) and the
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
`/health`. The always-warm base gears (primary, multimodal, embed, rerank) are
queried via `lobes fleet status --json`; the opt-in `minor` container is checked
directly via `docker inspect` because `fleet_containers()` only covers the
always-warm base set.

**Maps to t9 acceptance criterion:** all five co-resident gears fit on the GB10
with the documented budget and start successfully.

### B — Memory budget (informational)

Runs `nvidia-smi` and prints live GPU memory. Compare against the documented
budget:

| Gear | Tier | `--gpu-memory-utilization` | Approx GiB |
|---|---|---|---|
| primary (27B) | `main` | 0.45 | ~56 |
| multimodal (12B) | `multimodal` | 0.12 | ~15 |
| minor (4B, opt-in) | `minor` | 0.10 | ~13 |
| embed (0.6B) | — | 0.06 | ~7 |
| rerank (0.6B) | — | 0.06 | ~7 |
| **Default total** (without minor) | | **0.69** | **~85 / 128 GB** |
| **Full total** (with minor) | | **0.79** | **~98 / 128 GB** |

Non-fatal: nvidia-smi parse quirks do not block the exit code.

### C — Tier alias routing

POSTs `model=multimodal` to `POST /v1/chat/completions` and asserts **both**:

- `X-Lobes-Tier: multimodal` response header
- `body.model` contains `gemma-4-12B`

Smoke-tests `model=minor` (expect `X-Lobes-Tier: minor`) and `model=main`
(expect `X-Lobes-Tier: main`). The back-compat aliases (`normal`, `cheap`, `hard`)
are also accepted and emit the new-vocabulary tier in the response header
(`multimodal`, `minor`, `main` respectively).

**Maps to t9 acceptance criterion:** callers address the generate lane by
capability-tier alias and the gateway routes to the correct gear.

### D — Pressure backpressure (simulated)

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
   `docker-compose.yml`, **plus the audio overlay when scaffolded**, plus your
   own `docker-compose.override.yml` when present — with the threshold override
   merged on top of all of them via explicit `-f`, so the gateway keeps its full
   config (including `/v1/audio/*` routing and your local customisations) and
   only the threshold changes. The threshold override is merged **last** on
   purpose: it must win over an operator override that sets the same key, or
   this check would assert against your value instead of the lowered one.
   The vLLM backends are untouched. The gateway re-imports
   `lobes.gateway._pressure_policy` at container start, picking up the lowered
   threshold.
4. POST `model=main` and assert the request is **shed** (not substituted):
   HTTP `429`, a `Retry-After` header, and `X-Lobes-Tier-Reason: busy`.

The `LOBES_SWAP_DEGRADED_THRESHOLD` constant is read at **module import time**
inside the gateway container. Changing it requires a gateway restart; the
vLLM backends do not need to restart.

**Why the trap matters:** the override file must be removed and the
gateway must be recreated at defaults before the script exits, or the fleet will
stay with a lowered threshold that sheds all `main`/`hard` requests with `429`
permanently.
The `trap '_cleanup' EXIT` in the script handles this unconditionally — even if
the script is killed with Ctrl-C or fails partway through D.

**Maps to acceptance criterion:** swap or iowait pressure above the busy
threshold sheds `main` (and `multimodal`) requests with a `429` busy response —
it does not substitute another model (issue #85).

### E — Manual override

With the same lowered threshold still active from D, POSTs `model=main` **with
`X-Lobes-Override: 1`** and asserts:

- `X-Lobes-Tier: main` (override bypassed the shed — served, not `429`)
- `X-Lobes-Tier-Reason: manual_override`

The gateway's truthy tokens for `X-Lobes-Override` are `1`, `true`, `yes`.

**Maps to t9 acceptance criterion:** operators can force the requested tier
regardless of pressure when they explicitly set `X-Lobes-Override`.

### F — Pressure delta evidence (informational)

Fires three completions against `minor` + `multimodal` (4B/12B, with override to
bypass the still-lowered threshold) and three against `main` (27B, with
override). `lobes status --pressure --json` is sampled before and after each
burst to capture `swap_used_percent` and `iowait_percent` deltas. The evidence
is printed to stdout and optionally written to `--out`.

The expected pattern: the `minor`/`multimodal` burst produces smaller or equal
swap/iowait delta compared with the `main` burst, reflecting that the 27B primary
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

minor+multimodal burst (4B/12B gears, 3 completions with X-Lobes-Override):
  swap_before:   22.4%
  swap_after:    22.6%
  iowait_before: 0.8%
  iowait_after:  0.9%

main burst (27B gear, 3 completions with X-Lobes-Override):
  swap_before:   22.6%
  swap_after:    24.1%
  iowait_before: 0.9%
  iowait_after:  1.2%
```

Larger deltas on the `main` burst confirm the 27B primary uses more memory
bandwidth — the motivation for routing routine tasks to the `minor` or `multimodal`
tiers.

## Troubleshooting

**Check A fails for minor:** confirm `COMPOSE_PROFILES=minor` is set in the
deployment `.env` and the fleet was started **after** that change. Profiles are
read at `docker compose up` time. The `multimodal` gear starts automatically
(no profile needed); if it fails, check that `MULTIMODAL_BASE_URL` is wired in
the gateway env.

**Check D fails (shed not firing):** the gateway container's
`LOBES_SWAP_DEGRADED_THRESHOLD` env var was not set. Confirm the
`docker-compose.validate-tiers.override.yml` was written to `$COMPOSE_DIR`
before the `--force-recreate` call and that `docker compose up` picked it up
(it is passed explicitly via `-f`). Check
`docker inspect model-gear-gateway | grep -A5 Env`.

**Check C body.model mismatch:** the multimodal gear may not have finished loading
(vLLM load takes several minutes). Wait until `lobes fleet status` shows
`running (healthy)` for `model-gear-vllm-multimodal`.

**Gateway not restored after failure:** if the script crashes before the trap
can run, delete the override file and recreate the gateway. Include the audio
overlay and your own override if your deployment uses them, matching how `lobes`
composes the fleet — any explicit `-f` stops compose from auto-discovering
`docker-compose.override.yml`, so it must be named here or your local config is
dropped from the recreated gateway:

```bash
rm ~/.model-gear/docker-compose.validate-tiers.override.yml
docker compose --project-directory ~/.model-gear \
  -f ~/.model-gear/docker-compose.yml \
  -f ~/.model-gear/docker-compose.audio.yml \
  -f ~/.model-gear/docker-compose.override.yml \
  up -d --no-deps --force-recreate gateway
# Omit any -f line whose file your deployment does not have (the audio overlay
# needs --audio; docker-compose.override.yml exists only if you wrote one).
# Simplest of all: `lobes fleet up --apply` recreates the fleet with the
# correct compose files.
```
