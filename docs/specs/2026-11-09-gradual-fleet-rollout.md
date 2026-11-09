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
- \[challenge / reversibility: lobes/templates/fleet/docker-compose.yml gateway build -> image lobes-gateway:latest\] Before rebuilding, apply tags the running gateway image as lobes-gateway:<old-version>; rollback recreates the gateway from that tag. Today the rebuild overwrites the only tag, the old image goes dangling, and any prune deletes the rollback target
  - instruction: fixture: after apply, docker images lists lobes-gateway:<old-version>; rollback uses it with no network
  - honesty: a rollback succeeds with the package index unreachable
- \[challenge / adjacent systems: probes of spark, spark2, thor, orin 2026-11-09\] Member addressing is explicit per member: SSH target and user (spark/spark2/thor/orin users differ), gateway port (8001 on spark and cultureos, 8000 elsewhere), and the lobes binary invoked by absolute path (~/.local/bin/lobes is NOT on the non-login SSH PATH on 3 of 4 members). The mesh name is not the hostname (mesh spark = host spark-f8a9)
  - instruction: plan shows each member's resolved ssh target, port and lobes path; a member with no resolvable target is listed as unreachable, never guessed
  - honesty: no member target is derived by guessing; an unresolvable member is reported, not skipped silently
- \[challenge / adjacent systems: member CLI vs gateway versions\] CLI and gateway versions are tracked separately per member (spark: CLI 0.90.0; thor: CLI 0.89.0 with gateway 0.92.0.dev694 from a dev index). Plan shows both; apply refuses a downgrade (target older than the running gateway, PEP 440 ordering) unless explicitly allowed, and preflight honours `LOBES_DEV_INDEX_URL` for dev targets
  - instruction: unit-test version ordering incl. 0.92.0.dev694 < 0.94.0 and a refused downgrade
  - honesty: a downgrade never happens without an explicit flag
- \[challenge / security: thor has no `GATEWAY_API_KEY` in .env; cultureos key accepted by others\] Verification requests use each member's own gateway credential read over SSH, never the operator's by assumption; no key, mesh key or secret is ever printed in plan/apply/status output or the journal
  - instruction: grep the journal and captured stdout of a fixture run for the injected key value: zero hits
  - honesty: no secret value appears in any rollout output or file
- \[challenge / concurrency: no lock in lobes/runtime or lobes up\] apply takes a per-member lock on the member before its first write and refuses a member whose lock is held (another rollout, or another operator mid-step); the lock is released on success, failure and interrupt
  - instruction: two concurrent applies on one fixture member: exactly one proceeds
  - honesty: two rollouts can never write to the same member at the same time
- \[challenge / operations: docs/contracts/machine-cli-window.md\] apply refuses to recreate a member's gateway while that member has an active benchmark window (machine CLI 'window show'), unless explicitly overridden
  - instruction: fixture with an active window: member skipped with the reason
  - honesty: no gateway is recreated during an active benchmark window without an explicit override
- \[challenge / failure modes: operator box is itself a member (cultureos)\] The operator's own box is upgraded last in the rollout, after every remote member verifies, because upgrading its CLI replaces the binary that is running the rollout
  - instruction: plan orders the local member last regardless of canary choice
  - honesty: the rollout never replaces its own running binary before the remote members are done
- \[challenge / observability: c14 status needs a source\] Every apply writes an append-only journal on the operator box (member, step, before/after versions, verification result, timestamps); status reads it, and a rollout is resumable from it after an interrupt
  - instruction: kill apply mid-wave; status shows the interrupted member and re-running apply resumes there
  - honesty: an interrupted rollout can be resumed and status never reports a member as done that did not verify
- \[challenge / adjacent systems: lobes/runtime/`_lock.py` tracks the .env digest; deployments/ holds jetson-agx-orin and jetson-agx-thor locks\] A member with a deployment.lock.toml will report `lock_drift` after the repin; plan flags such members and apply reports the stale lock (re-capture stays the operator's call - there is no capture CLI)
  - instruction: fixture member with a lock: plan lists it under lock-affected members
  - honesty: every lock-affected member is named in plan output before any write

## Honesty conditions

- the announced flow exists as lobes rollout verbs and was used for at least one real release across the live mesh
- the operator can run it from any one member (including a gateway-only box like cultureos) without logging into the others by hand
- the before-state is evidenced by today's session (scope s1-s5), not reconstructed
- the drift and silent-failure examples are real: versions and the stt-en 404 were observed on 2026-11-09
- plan/apply/status are the whole surface; no step requires a hand-written script
- no rollout code path writes vLLM lane images, checkpoints, shape or profile files
- measured on the next real release, not a fixture
- both cases are covered by tests that fail on today's code
- a rollout between two adjacent released versions keeps every not-yet-upgraded member verified in the roster (no role drops to unverified because of the version mix)

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

## Assumptions

- \[challenge / distributed state: lobes/gateway/`_mesh_`\*.py has no wire-protocol version or compat check\] Consecutive lobes releases stay wire-compatible for mesh announce/verify, so old and new gateways coexist through the rollout window; features new in the target are reachable only from already-upgraded fronts (orin 0.89.0 could not route stt-en on 2026-11-09)

## Scope exploration

- `s1` — `2026-11-09 stt-lanes upgrade session`: Upgrading 4 gateways (cultureos 0.93.0, spark2 0.89.2, orin 0.89.0; thor on 0.92.0.dev694) to 0.94.0 took one hand-written SSH script per box; Claude Code's auto-mode blocked remote writes, so the operator ran each script
- `s2` — `publish.yml run 41882958341`: 0.94.0 merged 2026-11-04 but never reached PyPI: the test job hit 'No space left on device' and publish was skipped; nobody noticed for 5 days. After the rerun published it, pip inside python:3.12-slim still could not resolve lobes-cli==0.94.0 for several minutes (CDN lag), so the first gateway build failed
- `s3` — `lobes/templates/fleet/docker-compose.yml gateway service`: The gateway takes an explicit environment: list, never `env_file`. spark2's hand-kept 0.89-era compose lacked `STT_EN_BASE_URL`/`STT_EN_FEASIBLE`, so after upgrading to 0.94.0 the stt-en lane was silently infeasible until four lines were added to docker-compose.override.yml
- `s4` — `lobes doctor --repin-version --apply`: Exits non-zero while `gateway_version_match` FAILs, which is the very skew it is about to fix; a set -e upgrade script stopped there on spark2
- `s5` — `lobes up gateway --build --apply`: Single-box re-image path that works; a failed build leaves the old gateway running (safe). It does not wait for health or verify capabilities
- `s6` — `issue #262 and #198`: \#262: fleet up starts nothing when `MODEL_GEAR_VERSION` is unpublished (overlaps the preflight need). #198 'mass-update' is rollout-cli's cross-REPO code propagation, not box deployment; no existing fleet-rollout issue
- `s7` — `lobes/gateway mesh (GET /mesh/roster, /capabilities)`: GET /mesh/roster lists every member (name, origin, verified, roles) but carries NO version; each gateway's own unauthenticated GET /health returns {"version": ...} (issue #99), and /capabilities gives the per-role served state, so roster + per-member /health + /capabilities together give the before/after diff for verification
- `s8` — `challenge pass / reversibility lens: lobes/templates/fleet/docker-compose.yml gateway service + docker images on cultureos`: gateway image is lobes-gateway:latest; rebuild overwrites it, so h15 (previous image kept) does not hold today
  - seeds: `c20`
- `s9` — `challenge pass / security lens: GATEWAY_API_KEY presence per member`: thor .env has no `GATEWAY_API_KEY`; keys are per-box and must not be assumed or printed
  - seeds: `c23`
- `s10` — `challenge pass / distributed-state lens: lobes/gateway/_mesh_roster.py, _mesh_routing.py, _mesh_wire.py`: no wire-protocol version or compat negotiation found by grep; mixed-version coexistence is implicit (worked 0.89 to 0.94 on 2026-11-09)
  - seeds: `c24`
- `s11` — `challenge pass / concurrency lens: lobes/runtime/_compose.py, lobes/cli/_commands/up.py`: no lock file or flock around compose actions
  - seeds: `c25`
- `s12` — `challenge pass / operations lens: docs/contracts/machine-cli-window.md, lane-safety-runtime.md`: benchmark window and boot hold exist; a gateway recreate during a window was not covered by the spec
  - seeds: `c26`
- `s13` — `challenge pass / lifecycle lens: operator box as member`: cultureos runs the rollout and is itself a member; self-upgrade ordering was missing
  - seeds: `c27`
- `s14` — `challenge pass / observability lens: spec c14 rollout status`: status had no defined state source and no resume story
  - seeds: `c28`
- `s15` — `challenge pass / adjacent-systems lens: lobes/runtime/_lock.py, deployments/`: the lock tracks the .env digest, so a repin produces `lock_drift` on orin/thor variations
  - seeds: `c29`
- `s16` — `challenge pass / unexamined surfaces`: NOT examined: non-mesh single-box deployments (parked v1), per-box sidecars across a gateway recreate such as lobes-cloudflared (parked v2), macOS/non-uv installs, gateway-only boxes other than cultureos
- `s17` — `challenge pass / adjacent-systems lens: ssh probes of spark, spark2, thor, orin`: users, ports (8000/8001), hostnames vs mesh names, and lobes install path differ per member; CLI and gateway versions diverge (spark CLI 0.90.0, thor CLI 0.89.0 + gateway dev694); addressing source is open question q2
  - seeds: `c21`, `c22`

## Decisions

- Transport is SSH from the operator box to each member (user decision 2026-11-09); no admin endpoint on the gateway
- Compose drift (gateway env keys the target template passes but the member's deployed compose does not): plan reports it; apply patches docker-compose.override.yml only behind an explicit flag, after a backup, printing the diff (user decision 2026-11-09)
- On a failed post-upgrade verification: halt the rollout at that member, print the failed check and the exact rollback command; never auto-rollback (user decision 2026-11-09)
- Member addressing comes from an operator-typed rollout members file on the operator box (SSH target, user, gateway port, lobes path), never derived from the roster; plan reports roster members missing from the file as unreachable (user decision 2026-11-09, consistent with #92)

## Open parks

- [unknown_nonblocking] Single-box deployments with no mesh (no roster) - whether rollout should support them, or they keep using doctor --repin-version + lobes up gateway
- [unknown_nonblocking] The cloudflared tunnel container on cultureos (lobes-cloudflared) and other per-box sidecars were not examined for behaviour across a gateway recreate
