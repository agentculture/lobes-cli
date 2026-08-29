# Build Plan — deployment lock per box

slug: `deployment-lock-per-box` · status: `exported` · from frame: `deployment-lock-per-box`

> Every box's deployment is a committed lock: the compose files, overrides and Dockerfiles it actually runs are versioned in the repo under deployments/<box>/, secrets stay in the never-committed .env, lobes init --from-lock restores a box from it, and lobes doctor reports `lock_drift` when the deployed files no longer match.

## Tasks

### t1 — Positional gitignore rule + goldens negation

- covers: c22, h17, c23, h18, c8, h5
- acceptance:
  - git check-ignore proves .env, .cf-tunnel.env and .secrets.env ignored; .env.example and .env.sample not
  - a newly created tests/goldens/<new>.env and tests/goldens/shapes/<new>.env are both stageable
  - the test fails if the negation line is removed

### t2 — Variation identity resolver (machine type, never hostname)

- covers: c16, h12
- acceptance:
  - two hosts of the same machine type resolve to the same variation id
  - no resolved id contains a hostname, asserted against a fake host with a distinctive name
  - an unrecognised card resolves to an explicit unknown, never a guess

### t3 — CI secret gate over every committed deployment artifact

- covers: c9, h6, c26, h21, c32, h27
- acceptance:
  - a PR planting a token in a committed docker-compose.override.yml fails a REQUIRED job
  - the scanner path list names the lock and all verbatim-committed compose/Dockerfiles, not just repo defaults
  - the job fails on the planted token and passes on the clean tree

### t4 — Secret env family: second `env_file` entry across fleet services

- covers: c20, h15, c21, h16
- acceptance:
  - a deployment with no .secrets.env behaves byte-identically — compose config output unchanged
  - every fleet service that reads .env also reads the second file, verified by parsing the template
  - a fresh box reaches serving state with no hand-typed secret: keys generated or file-supplied

### t5 — Wheel-exclusion test for the deployments tree

- covers: c11, h8
- acceptance:
  - uv build produces a wheel whose namelist contains no deployments/ path
  - the test fails if packages= is widened to include it

### t6 — Lock writer: allowlist from `profile_env` into deployment.lock.toml

- covers: c24, h19, c25, h20, c17, h13, c7, h4
- acceptance:
  - feeding a .env containing `GATEWAY_API_KEY`, `HF_TOKEN` and `PRIMARY_PEER_ORIGIN` yields a lock containing none of them
  - the key set derives from `profile_env`, not a denylist: a newly added secret key stays excluded with no code change
  - the lock filename matches neither the .env suffix nor the .env. prefix rule
  - lock contents for a (profile, shape) agree with the matching tests/goldens file on every key both carry

### t7 — lobes init --from-lock: verbatim materialise, bypass resolution, guard machine type

- depends on: t6, t2
- covers: c1, h1, c13, h9, c4, h3, c3, h2, c36, h30
- acceptance:
  - a restored box has a byte-identical compose/override/Dockerfile set, proven by diff, with .env untouched
  - restoring on a box whose DETECTED card differs from the lock refuses by default; any override is explicit
  - a lock captured on a csv-mode card restores both GPU overlays; one from a devices card restores neither
  - profile/shape resolution is provably not consulted — patched to raise, the restore still succeeds
  - dry-run by default, --apply to write, matching the repo mutation-safety convention

### t8 — doctor `lock_drift` + switch staleness warning

- depends on: t6
- covers: c10, h7
- acceptance:
  - `lock_drift` names the specific differing files, not merely that drift exists
  - doctor --fix still never rewrites an existing .env line
  - running lobes switch makes the lock stale and `lock_drift` reports it; switch itself warns the lock needs re-committing

### t9 — Variation catalog: deployments/<id>/ layout + info file contract

- depends on: t6, t2
- covers: c14, h10, c15, h11, c19, h14
- acceptance:
  - a variation this box does not run is materialisable by --from-lock on a machine that never served it
  - every info file names an existing docs/evidence/ path or states no measured result; a test asserts cited paths exist
  - a variation with no transcript renders an explicit 'no measured result', never a blank readable as a measurement

### t10 — Buildability guard for a variation's pinned `MODEL_GEAR_VERSION`

- depends on: t6
- covers: c34, h28
- acceptance:
  - a variation whose gateway wheel is no longer installable is reported unbuildable with a clear message
  - the check fails early, not deep inside a docker build

### t11 — Secret rotation and revocation path

- covers: c35, h29
- acceptance:
  - the procedure names every place a copy of an inbound key lives, including each peer's <PREFIX>`_PEER_API_KEY`
  - a leaked-key drill is followable end to end by someone who did not write it
  - it states that git rm does not remove a committed secret and names the history-rewrite step

### t12 — Docs + framing: audience, before/after state, why it matters, success signals

- depends on: t7, t8, t9, t3
- covers: c27, h22, c28, h23, c29, h24, c30, h25, c31, h26
- acceptance:
  - each named audience reaches the artifact by a real mechanism, none served only by prose
  - a restored box serves the same model at the same knobs as before the restore
  - the motivating 2026-08-25 Spark/Thor divergence is cited to a transcript or issue, not remembered
  - each of the three costs is separately observable, and each success signal FAILS on a deliberately broken input

## Risks

- [unknown_nonblocking] the catalog's executable value is unproven: c34 shows a variation can outlive its buildable wheel and c36 shows a cross-machine-type restore is hazardous, so adoption by a third party may work only for variations whose wheel still resolves
- [unknown_nonblocking] no trust model exists for adopting a third party's variation — it names build contexts, bind mounts and image refs, so adoption is execution rather than reading; the challenge pass flagged this surface as unexamined
- [follow_up] t1 changes .gitignore for the whole repo; if the goldens negation is wrong in a way the test misses, a future golden silently fails to stage — merge t1 before any task that adds a committed artifact
