# The variation catalog

This directory publishes **docker variations** — complete, committed
deployments anyone can adopt. Pick one, restore it with `lobes init
--from-lock`, and you get the compose files, overrides and Dockerfiles that
box actually runs, byte for byte, without re-applying a single hand edit from
memory.

It is a catalog, not a backup: variations are published for machine types this
operator may not run, so a third party can adopt a working deployment without
owning the box it was measured on.

## Status — no real box has been captured yet

**This catalog is empty of real variations today.** Capturing one requires
running `lobes` on the physical hardware (a DGX Spark, a Jetson AGX Thor, a
Jetson AGX Orin) and committing what it actually serves. Nothing here is a
placeholder for a capture that "will look roughly like this" — an invented
variation would be exactly the manufactured evidence this contract exists to
prevent.

The layout, the info-file contract and the validator are in place and
exercised against clearly-labelled fixtures under
`tests/fixtures/deployments/`, so the first real capture is checked the moment
it lands (`tests/test_variation_catalog.py`).

## Layout

One directory per variation, directly under this one:

```text
deployments/
  README.md                     # this file
  VARIATION.template.md         # copy this when adding a variation
  <variation-id>/               # e.g. `spark`, or `spark__spark-lobe`
    deployment.lock.toml        # the committed lock — rendered knobs + digests
    VARIATION.md                # the info file (contract below)
    docker-compose.yml          # …and every other file the lock names,
    docker-compose.override.yml #    committed verbatim
    docker-compose.shape.yml
    Dockerfile.gateway
```

The directory **name** is the variation id — machine type or setup, never a
hostname (`lobes/variation.py`) — optionally suffixed `__<shape>` when a
deployment shape is applied. Both halves are cross-checked against the lock's
own `variation` and `shape` fields, so a directory cannot quietly describe a
different box than its lock does.

## What a variation must carry

`lobes/variation_catalog.py` enforces all of this, and
`tests/test_variation_catalog.py` runs it over every published variation:

- a `deployment.lock.toml` that loads, whose `[files]` table is **non-empty**
  and names every file needed to restore the deployment;
- every named file present, with bytes matching the digest the lock records;
- no unrecorded compose file or Dockerfile sitting beside them;
- a `VARIATION.md` with a title, a `## What this variation is` section, and a
  `## Measured result` section obeying the rule below.

## The info-file contract — evidence is cited, or absent out loud

A variation's `## Measured result` section does exactly one of two things:

1. **Cites** one or more transcripts under `docs/evidence/`, by path. It cites
   rather than restates the numbers — the transcript is the measurement, and a
   second copy of the figures would drift from it. Every cited path must
   resolve, and the lock's own `evidence` field must be among them.
2. **States, verbatim, `No measured result.`** — and the lock's `evidence`
   field is then empty.

Never both, and never neither. A blank section is a **failure**, not a
default: CLAUDE.md's issue #108 rule keeps a shape DECLARED until an
acceptance transcript lands (all four Orin shapes and `thor-muse` have none
today), so most of this catalog will honestly read "no measured result", and
publishing it must not manufacture the appearance of evidence.

## Adding a variation

1. Copy `VARIATION.template.md` to `deployments/<variation-id>/VARIATION.md`
   and fill it in.
2. Capture the lock from the deployment the box actually runs, and copy in
   every file the lock names.
3. Run the tests and the secret gate:

   ```sh
   uv run pytest tests/test_variation_catalog.py -q
   uv run python scripts/scan_deployment_secrets.py --root .
   ```

## Secrets are never here

The lock is an **allowlist** of rendered knob keys, not a redacted copy of a
deployed `.env` (`lobes/runtime/_lock.py`), so a credential cannot reach it by
someone forgetting to blank a line. The verbatim-committed compose files and
Dockerfiles get no such protection by construction, which is why
`scripts/scan_deployment_secrets.py` scans them in CI. Secrets live only in a
gitignored `.env`, which a restore never writes and never reads.
