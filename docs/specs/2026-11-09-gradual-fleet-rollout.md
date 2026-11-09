# gradual fleet rollout

> lobes rolls a new version out across the mesh gradually from one CLI: it checks the release is installable, upgrades one member per wave, verifies each through the mesh, and stops (or rolls back) on the first regression
> instruction: deliver as 'lobes rollout plan|apply|status|rollback' verbs; accept with a docs/evidence/ transcript of one real release rolled across the live mesh

## Audience

- The operator of a lobes mesh (and the agent driving lobes for them) who needs to move every member's gateway/CLI to a new lobes-cli release
  - instruction: check the spec's usage examples are written for the operator running lobes from one member box

## Before → After

- Before: Upgrading the mesh is per-box hand work: SSH to each member, uv tool install, doctor --repin-version, lobes up gateway --build, then hand-test. On 2026-11-09 that took four ad-hoc scripts and hit an unpublished release, PyPI CDN lag, a doctor exit code, and a compose that silently dropped a new lane's env keys
- After: One CLI flow - lobes rollout plan / apply / status - takes a target version, preflights it, upgrades members in waves (canary first), verifies each member before the next wave, and stops with the member, the failed check and a rollback path on the first regression

## Why it matters

- Each member upgraded by hand drifts (0.89.0 / 0.89.2 / dev694 / 0.93.0 coexisted today) and a half-done upgrade fails silently (stt-en infeasible with HTTP 404, not an error); a gradual, verified rollout turns that into a reported, stoppable sequence

## Requirements

- Preflight before touching any member: lobes-cli==<version> must pip-resolve inside the gateway's own base image (python:3.12-slim), retried with backoff to ride out index/CDN lag; refuse the rollout if it never resolves
  - instruction: unit-test the preflight with a fake index that 404s then serves; live: point at an unpublished version and confirm no member is touched
  - honesty: the preflight resolves from the same index and base image the gateway Dockerfile uses, so a pass predicts the build
- plan is read-only and derived from the mesh: roster members + each member's GET /health version + /capabilities, plus per-member drift (gateway env keys the target template passes that the member's deployed compose does not)
  - instruction: plan against the live mesh must list cultureos/spark/spark2/thor/orin with their current versions; a fixture compose missing `STT_EN_BASE_URL` must be reported as drift
  - honesty: plan performs zero writes on any member
- apply upgrades one wave at a time (default: one member per wave, a member that hosts nothing first as canary), each member step idempotent: backup .env, install the CLI, repin `MODEL_GEAR_VERSION`, rebuild only the gateway, wait for health; re-running a finished step is a no-op
  - instruction: run apply twice on the same member; the second run changes nothing and exits 0
  - honesty: a member is never left without a running gateway: a failed build keeps the old one (as lobes up gateway --build does today)
- After each member: verify version on /health, diff /capabilities before vs after (no role that was feasible/ready may become infeasible/not-ready), and send one real request per role the member hosts or forwards; halt the rollout on any regression
  - instruction: inject a regression (e.g. a member whose compose drops a lane key) in a fixture and confirm apply halts at that member and names the role
  - honesty: verification exercises the request path through the mesh, not only docker ps or /health
- Rollback per member restores the backed-up .env and recreates the gateway from the previous image; on a failed verification apply halts and prints the exact rollback command, which the operator runs (dry-run unless --apply)
  - instruction: after a forced failed verify, rollout rollback --member X --apply returns /health to the previous version
  - honesty: the previous gateway image is kept (tagged) so rollback needs no index access, and rollback is never triggered without an operator command
- Dry-run by default; --apply to commit, like every lobes write verb; rollout status reports per-member target/current version and last step result
  - instruction: lobes rollout apply without --apply prints the plan and touches nothing
  - honesty: no write happens without --apply

## Honesty conditions

- the announced flow exists as lobes rollout verbs and was used for at least one real release across the live mesh
- the operator can run it from any one member (including a gateway-only box like cultureos) without logging into the others by hand
- the before-state is evidenced by today's session (scope s1-s5), not reconstructed
- the drift and silent-failure examples are real: versions and the stt-en 404 were observed on 2026-11-09
- plan/apply/status are the whole surface; no step requires a hand-written script
- no rollout code path writes vLLM lane images, checkpoints, shape or profile files
- measured on the next real release, not a fixture
- both cases are covered by tests that fail on today's code

## Success signals

- Re-running today's upgrade (4 members to 0.94.0) is 1 plan + 1 apply command with 0 hand-written scripts, and every member ends at the target version with 0 roles regressed in /capabilities
  - instruction: repeat on the next release and record the transcript under docs/evidence/
- A rollout to an unpublished version touches 0 members, and a member whose deployed compose lacks a gateway env key the target needs is flagged before its upgrade (100% of today's stt-en-style gaps caught)
  - instruction: fixture tests for both cases

## Scope / boundaries

- Scope is the lobes gateway image + lobes CLI version on mesh members (`MODEL_GEAR_VERSION`, the gateway service). It does not swap model checkpoints, change vLLM lane images, or re-render shapes/profiles

## Non-goals

- Not a general config-management or fleet orchestrator: it never rewrites hand-kept compose files on its own; it reports drift and, at most, applies a narrow, backed-up, operator-approved patch
- Not cross-repo code propagation (that is rollout-cli / mass-update, issue #198)

## Scope exploration

- `s1` — `2026-11-09 stt-lanes upgrade session`: Upgrading 4 gateways (cultureos 0.93.0, spark2 0.89.2, orin 0.89.0; thor on 0.92.0.dev694) to 0.94.0 took one hand-written SSH script per box; Claude Code's auto-mode blocked remote writes, so the operator ran each script
- `s2` — `publish.yml run 41882958341`: 0.94.0 merged 2026-11-04 but never reached PyPI: the test job hit 'No space left on device' and publish was skipped; nobody noticed for 5 days. After the rerun published it, pip inside python:3.12-slim still could not resolve lobes-cli==0.94.0 for several minutes (CDN lag), so the first gateway build failed
- `s3` — `lobes/templates/fleet/docker-compose.yml gateway service`: The gateway takes an explicit environment: list, never `env_file`. spark2's hand-kept 0.89-era compose lacked `STT_EN_BASE_URL`/`STT_EN_FEASIBLE`, so after upgrading to 0.94.0 the stt-en lane was silently infeasible until four lines were added to docker-compose.override.yml
- `s4` — `lobes doctor --repin-version --apply`: Exits non-zero while `gateway_version_match` FAILs, which is the very skew it is about to fix; a set -e upgrade script stopped there on spark2
- `s5` — `lobes up gateway --build --apply`: Single-box re-image path that works; a failed build leaves the old gateway running (safe). It does not wait for health or verify capabilities
- `s6` — `issue #262 and #198`: \#262: fleet up starts nothing when `MODEL_GEAR_VERSION` is unpublished (overlaps the preflight need). #198 'mass-update' is rollout-cli's cross-REPO code propagation, not box deployment; no existing fleet-rollout issue
- `s7` — `lobes/gateway mesh (GET /mesh/roster, /capabilities)`: GET /mesh/roster lists every member (name, origin, verified, roles) but carries NO version; each gateway's own unauthenticated GET /health returns {"version": ...} (issue #99), and /capabilities gives the per-role served state, so roster + per-member /health + /capabilities together give the before/after diff for verification

## Decisions

- Transport is SSH from the operator box to each member (user decision 2026-11-09); no admin endpoint on the gateway
- Compose drift (gateway env keys the target template passes but the member's deployed compose does not): plan reports it; apply patches docker-compose.override.yml only behind an explicit flag, after a backup, printing the diff (user decision 2026-11-09)
- On a failed post-upgrade verification: halt the rollout at that member, print the failed check and the exact rollback command; never auto-rollback (user decision 2026-11-09)
