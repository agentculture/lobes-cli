# quick-wins + finish the rename

> lobes ships the #239 quick-wins batch and finishes the gear-to-lobes rename
> instruction: Ship as two PRs off main, each with its own version bump via .claude/skills/version-bump/scripts/bump.py, and follow the repo PR workflow: wait for Qodo/Copilot, reply to every thread, resolve all, never merge with unaddressed comments.

## Audience

- lobes operators running the three live boxes (DGX Spark, Jetson AGX Thor, Jetson AGX Orin), plus mesh consumers that read the gateway advert — and the next contributor, who currently meets two naming systems in one repo
  - instruction: No implementation. Use this to decide who the CHANGELOG entry and PR body are written for: operators upgrading three live boxes, not new installs.

## Before → After

- Before: Seven small defects sit in #239 with known file and line, each too small to justify its own PR and so none of them ships. Meanwhile the repo answers to two names: the CLI, verbs and JSON keys all say lobes, while every container, the log-wrapper scaffold and three .env keys still say model-gear — and lobes serve health-waits on one of those legacy names, so the fleet's own start verb reports failure on a stack that came up healthy
  - instruction: Verify each defect still reproduces at its cited line before writing the fix; drop any that is already fixed rather than reporting it fixed.
- After: PR1 lands the seven #239 quick wins. PR2 finishes the rename at depth (c): containers, constants, mg-logwrap.sh and the mount path carry lobes names, `MODEL_GEAR_VERSION` and `MODEL_GEAR_LOG_DIR` gain `LOBES_` primaries with read-both fallbacks, and the deprecated model-gear PyPI alias keeps publishing. One name, with no deployed box left stranded
  - instruction: Check this list against the merged tree at the end, item by item, and record the result in the delivery summary.

## Why it matters

- The rename is cheapest now and gets permanently more expensive later: deployments/ holds zero captured variations today, so the \[files\] digest churn costs nothing — after the first capture (#224) every renamed file drifts a real operator's lock with no verb to re-baseline it
  - instruction: Put the zero-captured-locks evidence in PR2's body as the stated reason the timing is right.

## Requirements

- The seven #239 items are all single-surface edits with a known file and line: fleet compose:1803 + env.example:369 (#158), serve.py:49 (#111), fleet compose:1032/1189/1297 (#120), docs/orin-profiles.md:33 (#171), tests/conftest.py (#172 leftover), init.py profile disclosure (#175 item 1), doctor.py `_version_skew_check` write path (#99)
  - instruction: Work the seven items in this order so the two same-file edits land as separate commits: (1) #158 fleet compose:1803 + env.example:369, (2) #120 fleet compose:1032/1189/1297, (3) #111 serve.py:49, (4) #172 tests/conftest.py autouse UNKNOWN pin, (5) #175 init.py profile-shadow disclosure line, (6) #99 named opt-in re-pin, (7) #171 docs/orin-profiles.md:33 + machine-profiles.md.
  - honesty: each of the seven items is verified against its cited file and line before the PR opens, not assumed from the issue text
- \#111 and the container-name rename touch the same surface: lobes/runtime/`_compose.py`:18 CONTAINER='model-gear-vllm' is both the legacy name serve.py:49 wrongly health-waits on AND one of the 18 model-gear-\* names the rename would change — they must be sequenced, not merged blindly
  - instruction: Fix serve.py:49 in PR1 against the CURRENT spelling: resolve the container from the deployment shape (fleet vs --single) and pass it to `wait_health`. In PR2, the constant rename then flows through that call site unchanged — do not revert or re-derive the PR1 fix.
  - honesty: \#111's fix is proven by a fleet lobes serve --apply that reports success, not only by a unit test asserting the container name string
- Renaming `MODEL_GEAR_VERSION` or `MODEL_GEAR_LOG_DIR` needs a read-both-keys fallback exactly like `LOBES_DIR`/`MODEL_GEAR_DIR` already has, because .env is merge-only by contract (#191): lobes init will never rewrite the old key on a deployed box, so a rename with no fallback silently drops the operator's value
  - instruction: Mirror the `LOBES_DIR` pattern already at `_compose.py`:324-336: read the `LOBES_` key first, fall back to the `MODEL_GEAR_` key, and name the legacy source in any error text the way '$`MODEL_GEAR_DIR` (legacy)' already does.
  - honesty: a box whose .env sets only the OLD key (`MODEL_GEAR_VERSION` / `MODEL_GEAR_LOG_DIR`) still resolves that value after upgrading, proven by a test that writes only the legacy key
- Renaming the 18 `container_name` entries changes every file digest the deployment lock tracks in \[files\] and, if the .env keys move too, the \[env\] allowlist derived from profiles/render.py — so every locked box reports `lock_drift` on upgrade and cannot restore with --from-lock until it re-captures, which #224 says there is no verb for
  - instruction: Before renaming, run scripts against a locally captured lock if one exists; if none does, record that fact in the PR body as the reason the \[files\] churn is safe to take now.
  - honesty: a lock captured before the rename and restored after it is either still restorable or fails with a named, actionable error — never a silent partial restore
- tests/`test_upgrade_compat.py` builds a previous-version scaffold from tests/fixtures/`upgrade_compat`/{single,fleet}/docker-compose.yml, both of which carry model-gear-\* container names — the rename must keep those fixtures spelling the OLD names, since their whole job is proving an old scaffold still works
  - instruction: Leave tests/fixtures/`upgrade_compat`/{single,fleet}/docker-compose.yml byte-identical. If `test_upgrade_compat` fails, the production code is wrong, not the fixture.
  - honesty: tests/fixtures/`upgrade_compat`/{single,fleet}/docker-compose.yml still spell the OLD model-gear-\* container names after PR2, and `test_upgrade_compat` passes
- PR1 ships the seven #239 quick wins and PR2 ships the rename on top; #111's serve.py:49 fix therefore lands in PR1 reading `_compose`.CONTAINER as it is spelled today, and PR2 rebases on it
  - instruction: Branch PR2 off PR1's head, not off main, so serve.py:49 is already fixed when the constants move.
  - honesty: PR2 rebases on PR1 rather than reverting it: the serve.py:49 fix survives the constant rename intact
- `MODEL_GEAR_LOG_DIR` becomes a read fallback, not a deletion: `LOBES_LOG_DIR` is the new primary and `_compose.py`:30 `LOG_DIR_ENV` grows a two-key read, because the sole-key check confirmed it is NOT already a fallback
  - instruction: Change `_compose.py`:30 `LOG_DIR_ENV` into a two-key read: `LOBES_LOG_DIR` first, `MODEL_GEAR_LOG_DIR` second. Update the compose default at templates/docker-compose.yml to ${`LOBES_LOG_DIR`:-${`MODEL_GEAR_LOG_DIR`:-./logs}} so a legacy .env still resolves.
  - honesty: `LOBES_LOG_DIR` wins when both keys are set, and `MODEL_GEAR_LOG_DIR` alone still resolves — both directions covered by a test
- The rename copies rather than moves at the FILE level, mirroring what c25 does at the env-key level: an existing deployment.lock.toml is copied forward to the new names rather than invalidated, and a renamed scaffold file leaves the old one in place rather than deleting it — so a box mid-upgrade resolves under either spelling and 'move' is never a moment where neither works
  - instruction: Copy, never rename in place. For deployment.lock.toml: read the old \[files\] keys, emit the new names alongside, keep the old entries resolvable. For mg-logwrap.sh: register the new name in `FLEET_TEMPLATES` and leave the old file on disk untouched (doctor already heals the new one, per c13). Deleting either old artifact is a separate, later decision — not this PR.
  - honesty: a deployment.lock.toml written before PR2 still restores after PR2 — proven against a fixture lock carrying the OLD file names, not only against a freshly written one
- BUILD-ARG SUBSTITUTION HAS NO PYTHON FALLBACK, so the version key FAILS CLOSED. docker-compose.yml:1717 passes the version as a build ARG and Dockerfile.gateway:26,28 runs 'pip install lobes-cli==${...}'; a merge-only .env on an upgraded box carries only the old key, and an empty interpolation would silently build 'pip install lobes-cli=='. Resolution: the compose/Dockerfile side uses `LOBES_VERSION` only and ERRORS when it is absent — no nested-default interpolation, which was never verified on the Jetsons' docker compose. lobes init's merge-only append is what supplies `LOBES_VERSION` on an upgraded box
  - instruction: Make the absent case a NAMED error, not an empty string: fail the build or the up with a message that says `LOBES_VERSION` is missing and to run 'lobes init --apply' to add it. Verify by building the gateway image against a .env carrying only `MODEL_GEAR_VERSION` — it must fail loudly, never install an unpinned wheel.
  - honesty: a box whose .env carries ONLY `MODEL_GEAR_VERSION` builds the gateway image successfully after PR2 — proven by an actual image build against a legacy-key-only .env, not by a Python unit test, because the failure is in compose interpolation and a Python test cannot see it
- `MG_LOG_DIR` and `MG_LOG_NAME` are a THIRD legacy prefix the frame never enumerated (17 uses each, lobes/templates). They are in-container env vars, not operator-typed .env keys, so they carry no merge-only risk and can be renamed freely with the mount path — but c7's four-surface enumeration is incomplete without them
  - instruction: Rename `MG_LOG_DIR`/`MG_LOG_NAME` and the /logs/<name> mount point in the same commit; existing per-boot logs under the old host path are left where they are and are not migrated.
  - honesty: `MG_LOG_DIR` and `MG_LOG_NAME` are renamed together with the /logs/model-gear mount path in one commit, and a lane started after PR2 writes its per-boot log where lobes logs looks for it
- lobes/explain/catalog.py is TAUGHT CONTENT shipped in the CLI and names the legacy containers directly (lines 137, 204, 280-282, 315) plus `MODEL_GEAR_VERSION` (547, 1505) — so 'lobes learn' and 'lobes explain' actively teach the old names and must be refreshed by PR2. The ('model-gear',) alias at line 1525 is a deliberate back-compat entry and STAYS
  - instruction: Refresh lobes/explain/catalog.py lines 137, 204, 280-282, 315, 547 and 1505 to the new names. Do NOT touch line 1525 — that alias is the deprecated dist/repo name and must keep resolving.
  - honesty: after PR2, 'lobes explain gateway' and 'lobes learn' name no legacy container, and the ('model-gear',) back-compat alias at catalog.py:1525 still resolves

## Honesty conditions

- PR1 and PR2 both merge with the version-check CI job green — every AgentCulture PR bumps the version, no exceptions
- the gears-first 3.67x counterpart is cited to docs/evidence/ or to the 2026-08-04 measurement, never restated from memory
- no existing .env line is rewritten by the #99 re-pin path; the write is reachable only through an explicitly named opt-in flag
- no file under docs/evidence/ or .devague/ differs after PR2 — verified by git diff --stat on those paths
- the publish workflow still builds and publishes the model-gear alias after PR2, verified by reading publish.yml, not assumed
- the rename ships mg-logwrap.sh's new name in `FLEET_TEMPLATES` in the SAME commit as the compose reference, and lobes doctor on an un-healed box names the missing file rather than failing opaquely
- pip install model-gear still resolves to lobes-cli after PR2
- the three boxes named are the three that actually run lobes today — verified against the deployment evidence under docs/evidence/, not assumed
- every defect named in the `before_state` is still open and reproducible at its cited file and line when PR1 opens; any that is already fixed is dropped from the batch rather than reported as fixed
- every claim in the `after_state` is checked against the merged tree, not against the plan
- the zero-captured-locks claim is re-checked at PR2 time, not carried over from the 2026-08-31 scan — if a lock has appeared since, the cheapest-now argument is withdrawn and c26's copy-forward path is exercised for real
- the git grep in this success signal is actually run and its output recorded in the PR body
- no code path added by PR2 writes, moves or requires migrating a deployment directory; a box whose deploy dir is still ~/.model-gear keeps working untouched
- for each of the three key pairs a test proves the old-key-only case still resolves, and no branch of the rename deletes a `MODEL_GEAR_` read
- each of the five numbers is actually measured and pasted into the PR body, not asserted — the grep, the diff --stat, the issue closures and the test count
- git diff --stat main -- CHANGELOG.md shows only the new release entry after each PR, never edits to historical entries
- after PR2, running scripts/live-check.sh and scripts/accept-shape.sh against a real box succeeds — the seven scripts are exercised, not just grepped
- with no <ROLE>`_CONTAINER_NAME` set anywhere, a re-up on an existing deployment reports Running (not Recreated) for every lane — verified against a real box, not only the busybox probe
- a box upgraded with lobes init --apply keeps every container it was already running: docker ps shows the same names and the same uptimes before and after

## Success signals

- lobes serve on a fleet deployment waits on a container that exists and reports success; git grep for model-gear outside docs/evidence, .devague and tests/fixtures/`upgrade_compat` returns only packaging/model-gear and the two back-compat fallback reads; the full suite passes including `test_upgrade_compat`; and a box upgraded without touching its .env still resolves its log dir and version pin
  - instruction: Run the grep as: git grep -l 'model-gear' -- . ':!docs/evidence' ':!.devague' ':!tests/fixtures/`upgrade_compat`' ':!CHANGELOG.md' and paste the output into the PR body.
- Measurable targets: 7 of 7 #239 items closed by PR1; 0 files changed under docs/evidence and .devague across both PRs; 0 occurrences of model-gear outside packaging/model-gear, tests/fixtures/`upgrade_compat`, docs/evidence, .devague and CHANGELOG.md after PR2; 3 of 3 env key pairs resolving from the legacy key alone in tests; and the full suite green with 0 failures including `test_upgrade_compat`
  - instruction: Run and paste: gh issue list --state closed for the 7; git diff --stat main -- docs/evidence .devague; the c23 grep; uv run pytest -n auto -q tail line.

## Scope / boundaries

- docs/orin-profiles.md and docs/machine-profiles.md are the ONLY surfaces #171 touches — it is a documentation correction of a measured figure, not a knob change; lobes/profiles/builtin/orin.toml is not edited
  - instruction: Edit docs/orin-profiles.md:33 to name the senses-first boot order beside the 802,644/6.12x figure and add the gears-first 480,431/3.67x counterpart; then strengthen the boot-ordering caveat in docs/machine-profiles.md to say order changes the served KV budget, not just OOM risk. Do not edit lobes/profiles/builtin/orin.toml.
- \#99's fix must not relax lobes doctor --fix's never-rewrite-an-existing-.env-line convention; re-pinning `MODEL_GEAR_VERSION` has to be an explicit named opt-in, because #174 (init --force destroyed 12 operator-typed keys) and #191 (init is merge-only always) are what that convention costs when broken
  - instruction: Add the re-pin behind its own named flag on doctor --fix (or lobes up gateway); assert in a test that a .env with an existing `MODEL_GEAR_VERSION` line is byte-identical after a --fix --apply run without that flag.
- docs/evidence/ transcripts are frozen records of what a box actually printed — the 10 evidence files containing `MODEL_GEAR_` must NOT be rewritten by the rename; the same holds for .devague/frames, .devague/plans and .devague/deliveries, which record what was decided at the time
  - instruction: After PR2's last commit, run: git diff --stat main -- docs/evidence .devague — it must print nothing. Never use a repo-wide sed.
- packaging/model-gear/ and .github/workflows/publish.yml:122-127 keep publishing the deprecated model-gear PyPI alias that redirects to lobes-cli; finishing the rename must not unpublish or stop building it, because that breaks every existing pip install
  - instruction: Read .github/workflows/publish.yml:118-130 after the rename and confirm the model-gear alias job still builds; do not edit packaging/model-gear/.
- Only depth (d) is out of scope: packaging/model-gear/ and .github/workflows/publish.yml:122-127 keep publishing the deprecated model-gear PyPI alias that redirects to lobes-cli. Container names, the 14 `_compose.py` constants, mg-logwrap.sh and the /logs/model-gear mount path moved IN when v1 resolved to 'yes, rename'.
  - instruction: Confirm 'pip download model-gear' still resolves post-merge, or read publish.yml to confirm the job is unchanged.
- The rename touches env var NAMES only, never on-disk deployment DIRECTORIES: ~/.model-gear stays a valid legacy default alongside ~/.lobes, no deployment dir is migrated, moved or renamed, and no operator is asked to relocate one
  - instruction: Grep the diff for any write to a deployment directory path; there must be none. ~/.model-gear must remain a valid --compose-dir target after PR2.
- CHANGELOG.md is frozen historical record like docs/evidence and .devague: its model-gear occurrences describe releases that really were named that, and the rename must not rewrite them. c9 named evidence and .devague but omitted CHANGELOG, and c23's success-signal grep already excludes it — this closes that inconsistency
  - instruction: Add the new CHANGELOG entry via the version-bump script only; never sed historical entries. The c23 success grep already excludes CHANGELOG.md — keep that exclusion.

## Non-goals

- `MODEL_GEAR_DIR` is NOT part of the remaining rename work — lobes/runtime/`_compose.py`:303-336 already resolves --compose-dir then $`LOBES_DIR` then $`MODEL_GEAR_DIR` (legacy) with the legacy source named in the error text; that leg shipped and only its deprecation loudness is open
  - instruction: Read `_compose.py`:303-336 and change nothing there. Cite it as the reference pattern for c25's other two key pairs.
- PR2 does NOT ship the <ROLE>`_CONTAINER_NAME` override knob, and does NOT refactor the 14 hardcoded constants at `_compose.py`:18,48-65 into env-resolved reads. The mechanism was probed and works (absent key holds the default and a re-up prints 'Running', so an existing box could upgrade with zero downtime) but the operator's call is that the keys are not a requirement — PR2 hard-renames the defaults and accepts one restart per lane
  - instruction: Replace the 14 literals at `_compose.py`:18,48-65 with reads that resolve <ROLE>`_CONTAINER_NAME` from .env and fall back to the same default the compose template uses, so the two can never disagree. Derive both from ONE table the way `HAND_LORA_MODULES` is read by both the engine flag and the gateway alias.
- lobes init does NOT append legacy container-name overrides on a pre-existing deployment in PR2 — that migration belongs to the deferred override work, not here

## Assumptions

- The rename's real surface is four things at depth (c): 18 `container_name`: model-gear-\* in the templates (mirrored by CONTAINER plus 13 `FLEET_`\* constants in `_compose.py`), the bind-mounted mg-logwrap.sh scaffold file (39 refs), two operator-typed .env keys with no `LOBES_` successor (`MODEL_GEAR_VERSION` 162 refs, `MODEL_GEAR_LOG_DIR` 39 refs), and the /logs/model-gear in-container mount path. All four are now in scope.
  - instruction: Enumerate all four surfaces before editing: git grep -n '`container_name`: model-gear' lobes/templates/, the CONTAINER + `FLEET_`\* block at `_compose.py`:18-74, git grep -n mg-logwrap, and git grep -n '/logs/model-gear'.
- Renaming mg-logwrap.sh is LESS risky than first scoped: `LOG_WRAPPER` is a member of `_compose`.`FLEET_TEMPLATES` (`_compose.py`:162,173) and doctor.py:530 reads that dict, so lobes doctor already reports the file missing and lobes doctor --fix --apply heals it — the same mechanism #227 used for `qwen3_reranker`.jinja. The residual risk is ordering (a box that re-renders compose and starts it BEFORE healing gets a bind-mount to a file that is not there) and an orphaned old script left on disk, which is harmless.
  - instruction: Add the new wrapper name to `FLEET_TEMPLATES` in the same commit that changes the compose bind-mount reference; verify lobes doctor on a stale deployment dir names the missing file.
- Depth (b) still changes \[files\] digests, because Dockerfile.gateway, Dockerfile.realtime and Dockerfile.chatterbox all carry the renamed key in their lobes-cli== pip pin — but deployments/ ships only README.md and VARIATION.template.md, so ZERO real variations are captured and no operator hits `lock_drift` from this. That window closes the moment the first box is captured (#224), which argues for doing the rename now rather than after
  - instruction: Re-run the lock scan (see c22's honesty condition) at PR2 time and paste the result into the PR body.
- Seven scripts couple to the legacy names (accept-by-proxy.sh, accept-shape.sh, live-check.sh, `probe_reranker_calibration.py`, spec-arms.py, spike-preflight.sh, validate-tiers.sh). These are live operator/acceptance tooling, not frozen record, so they move WITH the rename — but they are not covered by the test suite, so a rename that misses one fails only at the next acceptance run

## Scope exploration

- `s1` — `lobes/templates/fleet/docker-compose.yml + env.example (#158, #120)`: three separate quick wins live in one file: `MULTIMODAL_BASE_URL`'s non-empty default at :1803 (env.example:369), and the dead `VLLM_ATTENTION_BACKEND` env at :1032/:1189/:1297 which the same file contradicts at :840 — they will collide in one PR and should be one commit each
  - seeds: `c2`
- `s2` — `lobes/cli/_commands/serve.py:49 and lobes/runtime/_compose.py:18,48-65`: serve.py calls `wait_health`(port) with the default container=`_compose`.CONTAINER = 'model-gear-vllm', the legacy single-model name; `_compose.py` also holds 13 more `FLEET_`\* constants all spelled model-gear-\*, so the #111 fix reads the same constants the rename rewrites
  - seeds: `c4`
- `s3` — `docs/orin-profiles.md:33 and docs/machine-profiles.md`: the 802,644-token / 6.12x senses figure is quoted with no boot order; the live 2026-08-04 measurement at identical knobs was 480,431 / 3.67x — the fix is annotation plus the gears-first counterpart, and does not touch lobes/profiles/builtin/orin.toml
  - seeds: `c3`
- `s4` — `lobes/cli/_commands/doctor.py _version_skew_check + init.py:756 + lobes/runtime/_env.py`: doctor already reads the gateway's deployed version off GET /health and init writes `MODEL_GEAR_VERSION` exactly once; the only missing piece is a WRITE, which lands squarely on the merge-only .env convention that #191 and #174 established
  - seeds: `c5`
- `s5` — `lobes/runtime/_compose.py:303-336 (deploy-dir resolution)`: `MODEL_GEAR_DIR` already has its `LOBES_DIR` successor with a documented three-step fallback and a '$`MODEL_GEAR_DIR` (legacy)' error label — this leg of the rename is finished, and the 19 remaining `MODEL_GEAR_DIR` refs are the fallback itself plus its docs
  - seeds: `c6`
- `s6` — `git grep MODEL_GEAR_ (208 hits, 83 files)`: only three distinct identifiers exist — `MODEL_GEAR_VERSION` (162), `MODEL_GEAR_LOG_DIR` (39), `MODEL_GEAR_DIR` (19) — and they split cleanly by surface: shipped code and templates, versus frozen docs/evidence and .devague records that must not be touched
  - seeds: `c7`
- `s7` — `lobes/templates/*/docker-compose*.yml container_name entries + _compose.py:18,48-65`: 18 `container_name`: model-gear-\* lines in the templates are mirrored by CONTAINER plus 13 `FLEET_`\* constants in `_compose.py`; `_health` and lobes logs both resolve containers by these exact strings, so template and constant must move in the same commit or the health-wait breaks the way #111 already breaks it
  - seeds: `c11`
- `s8` — `lobes/runtime/_lock.py:89-103 (EXCLUDED_RENDERED_KEYS / EXCLUDED_KEY_SUFFIXES)`: the lock's \[env\] is an allowlist derived from profiles/render.py minus `COMPOSE_PROFILES` and any `_URL` suffix, and \[files\] is sha256 per file — a rename moves both, so every locked box drifts on upgrade with no capture verb (#224) to re-baseline it
  - seeds: `c11`
- `s9` — `tests/fixtures/upgrade_compat/{single,fleet}/docker-compose.yml + test_upgrade_compat.py`: both fixtures carry model-gear-\* container names and exist specifically to prove a previous-version scaffold still works; they are the one place the OLD spelling must survive the rename verbatim
  - seeds: `c12`
- `s10` — `mg-logwrap.sh (39 refs across templates, scripts and tests)`: a packaged scaffold file bind-mounted by the fleet compose — renaming it splits the compose reference from the on-disk file on any box that re-renders one without the other, the same failure shape as deployment-lock deviation d2 (#226)
  - seeds: `c13`
- `s11` — `packaging/model-gear/ + .github/workflows/publish.yml:122-127`: the deprecated model-gear PyPI alias is built by rewriting its pyproject version and lobes-cli== pin at publish time; it is a live install path for existing consumers and is out of scope for removal
  - seeds: `c10`
- `s12` — `docs/evidence/ (10 files), .devague/{frames,plans,deliveries} (13 files)`: these carry `MODEL_GEAR_` as frozen historical record — evidence transcripts are verbatim console output and devague records are what was decided at the time; a repo-wide sed would corrupt both
  - seeds: `c9`
- `s13` — `lobes CLI verb surface (lobes/cli/_commands/, 27 modules)`: no verb name, flag or JSON key contains 'gear' or 'model-gear' — the user-facing CLI was fully renamed already; what remains is deployment-artifact naming and internal constants, which is why 'finalize' is a smaller question than the 208 grep hits suggest
- `s14` — `CLAUDE.md deployment-model section`: the resolution order is documented as --compose-dir then $`LOBES_DIR` then ~/.lobes, falling back to legacy $`MODEL_GEAR_DIR` then ~/.model-gear 'so a pre-rename deployment keeps working' — that sentence is the repo's own stated policy on renames, and it argues for fallbacks over replacements
- `s15` — `lobes/profiles/render.py::profile_env vs lobes/cli/_commands/init.py:756`: `MODEL_GEAR_VERSION` is written by init, not rendered as a profile knob, so it never enters the lock's \[env\] allowlist — the depth-(b) key rename is invisible to that half of the lock
  - seeds: `c15`
- `s16` — `deployments/ (README.md + VARIATION.template.md only)`: the variation catalog holds ZERO captured boxes, so the \[files\] digest churn a rename causes is theoretical today; the cost becomes real only after the first capture, which #224 says has no verb yet
  - seeds: `c15`
- `s17` — `lobes/runtime/_compose.py:165-177 FLEET_TEMPLATES + doctor.py:530`: mg-logwrap.sh is already a registered scaffold template, so init writes it, doctor reports it missing and --fix --apply heals it — the rename inherits #227's proven `qwen3_reranker`.jinja mechanism rather than needing a new one
  - seeds: `c13`
- `s18` — `~/.lobes and ~/.model-gear on spark, thor and orin (live check, read-only)`: ZERO deployment.lock.toml exists on any of the three boxes — checked ~/.lobes, ~/.model-gear and a depth-4 find under $HOME on each, plus deployments/ in-repo which holds only README.md and VARIATION.template.md; so the rename's \[files\] digest churn drifts no real lock and hard question q2 is answered no
  - seeds: `c15`
- `s19` — `challenge pass / adjacent-systems lens: docker-compose.yml:1717 + Dockerfile.gateway:26,28`: the version key crosses a Python-to-compose boundary as a build ARG; read-both-keys in Python cannot cover it, and an empty interpolation yields 'pip install lobes-cli==' rather than an error
  - seeds: `c28`
- `s20` — `challenge pass / adjacent-systems lens: scripts/ (7 files)`: accept-by-proxy.sh, accept-shape.sh, live-check.sh, `probe_reranker_calibration.py`, spec-arms.py, spike-preflight.sh and validate-tiers.sh all reference legacy names and are NOT covered by the test suite
  - seeds: `c31`
- `s21` — `challenge pass / adjacent-systems lens: tests/goldens/`: CLEAN — no golden contains a container name (grep -rl model-gear tests/goldens returns nothing), so the goldens contract does not churn on the rename; c12's `upgrade_compat` coverage is sufficient
- `s22` — `challenge pass / failure-mode lens: live docker compose 29.1.3 probe in scratch space`: changed `container_name` on a running service and re-upped: compose matched by service label, Recreated, zero orphans — DISPROVES c14's manual-reap premise
  - seeds: `c32`
- `s23` — `challenge pass / unstated-assumptions lens: MG_LOG_DIR / MG_LOG_NAME (17 uses each)`: a third legacy prefix the frame never enumerated; in-container only, so no merge-only exposure, but c7's four-surface list was incomplete
  - seeds: `c29`
- `s24` — `challenge pass / reversibility lens: rollback to a pre-PR2 lobes-cli`: an operator who downgrades keeps `LOBES_LOG_DIR` in .env, which old code does not read, so the log dir silently reverts to the default rather than erroring — asymmetric with the forward path, which reads both keys
- `s25` — `challenge pass / operations lens: docs/evidence, .devague, CHANGELOG.md`: CHANGELOG.md carries model-gear as accurate release history but was NOT named frozen by c9, while c23's grep already excludes it — an inconsistency between two confirmed claims
  - seeds: `c30`
- `s26` — `challenge pass / security lens: .gitignore positional rule, .secrets.env, GATEWAY_API_KEY`: CLEAN — the rename touches no credential path: the positional gitignore rule keys on the .env suffix (unchanged), .secrets.env is not renamed, and no auth key is in the renamed set
- `s27` — `challenge pass / overlooked-actors lens: sibling repos (colleague, daria, reachy-mini-cli, eidetic-cli)`: NOT EXAMINED — the pass was scoped to this repo; these consumers address the gateway by role name and URL, which the rename does not touch, but no grep was run against their trees to confirm none reads a container name or `MODEL_GEAR_` key
- `s28` — `challenge pass / migration lens: live docker compose 29.1.3 probe of container_name interpolation`: `container_name`: ${X:-default} interpolates correctly; absent key holds the legacy default and a re-up is a no-op ('Running'), set key renames with no orphan — this is what makes the zero-downtime upgrade path real rather than hypothetical
  - seeds: `c33`
- `s29` — `challenge pass / overlooked-actors lens: ~/git on the Thor and the Orin (ssh, read-only)`: CLEAN — the Thor has zero legacy-name hits outside lobes-cli; every Orin hit is a lobes-cli worktree. Only edge-ai-lab/tests/`test_arm_run.py`:498 (writes `MODEL_GEAR_VERSION`=1) and an eidetic-cli comment couple cross-repo, both local and both survived by the read-both-keys fallback
- `s30` — `challenge pass / documentation lens: lobes/explain/catalog.py`: the CLI SHIPS the legacy names as taught content at lines 137/204/280-282/315/547/1505 — 'lobes learn' and 'lobes explain' actively teach them, a surface neither c7's four-surface enumeration nor c31's scripts list covered
  - seeds: `c36`

## Decisions

- PROBED, and the original premise was wrong: renaming `container_name` does NOT orphan containers. Live docker compose 29.1.3 matched the running container by its com.docker.compose.service label, printed 'Recreated', and left zero orphans — so there is no manual reap step and no port-holding conflict. The real and only cost is that each renamed lane RESTARTS once, which for a 27B NVFP4 lane means a model reload of minutes. That cost is accepted for PR2; the \*`_CONTAINER_NAME` override that would avoid it is deferred, not required (see the follow-up park)
  - instruction: Document the reap step in the PR body and CHANGELOG: after upgrading, 'docker rm' the orphaned model-gear-\* containers once the lobes-\* ones are healthy. Do not automate it.
- Every renamed env key follows one rule with no exceptions: the `LOBES_` name becomes primary, the `MODEL_GEAR_` name stays readable as a fallback, and nothing is deleted. `LOBES_DIR`/`MODEL_GEAR_DIR` already satisfies it (no change needed); `MODEL_GEAR_VERSION` and `MODEL_GEAR_LOG_DIR` are brought up to it
  - instruction: Implement all three the same way, mirroring `_compose.py`:324-336: try the `LOBES_` key, fall back to the `MODEL_GEAR_` key, and label the legacy source in error text as '$`MODEL_GEAR_X` (legacy)'. `LOBES_DIR`/`MODEL_GEAR_DIR` is already correct — read it as the reference implementation, do not rewrite it.
- PROBED AND DISPROVEN: c14's premise is wrong. A live docker compose 29.1.3 probe changed only `container_name` on a running service and ran 'docker compose up -d': compose matched the existing container by its com.docker.compose.service label, printed 'Recreated', started it under the new name, and left ZERO orphans (docker ps -a showed only the new name). Renaming `container_name` therefore needs no manual reap step and creates no port-holding orphan — c14 should be rejected or amended to say the opposite
- The container rename is a HARD rename of the defaults in PR2: templates and constants both move to lobes-\*, each lane restarts once, and no override knob ships. The default-renamed-plus-overridable design is probed and viable but deferred to a later refactor — recorded so the next person does not re-derive it

## Hard questions

All three were answered during the /think interrogation and the /challenge
pass on 2026-08-31. They are kept with their answers rather than dropped,
because every one of them changed a claim. **Note for re-exports:** `devague
export` regenerates this section from frame state and drops these answers —
re-apply them after any re-export.

- **Is there any box in the mesh with a captured `deployment.lock.toml` today?**
  **ANSWERED — no.** Checked live and read-only on all three boxes: `~/.lobes`,
  `~/.model-gear` and a depth-4 `find` under `$HOME` on the DGX Spark (local)
  and the Jetson AGX Thor and Orin (ssh). Zero locks anywhere; `deployments/`
  holds only `README.md` and `VARIATION.template.md` (`s18`). This is the whole
  basis of `c22`, so `h22` requires re-running the scan at PR2 time rather than
  trusting this result.
- **Does renaming `mg-logwrap.sh` require a coordinated scaffold step, or does
  `lobes doctor --fix --apply` already heal it?** **ANSWERED — doctor already
  heals it.** `LOG_WRAPPER` is in `_compose.FLEET_TEMPLATES` (`_compose.py`:162,173)
  and `doctor.py`:530 reads that dict, so `init` writes it, `doctor` reports it
  missing and `--fix --apply` heals it — the mechanism #227 used for
  `qwen3_reranker.jinja` (`s17`). This **downgraded** the risk: `c13` originally
  called this the riskiest single item and was amended to say the opposite.
  Residual risk is ordering only, plus a harmless orphaned old script.
- **Does `lobes init` reliably distinguish a pre-existing deployment from a fresh
  scaffold, or does `--force` blur that?** **ANSWERED — moot for PR2.** `c34` is
  now a non-goal: no container-name override is appended, so `init` never needs
  to make that distinction. The question returns intact if the deferred override
  work (`v5`) is picked up, and it is the main thing that could sink `c34` if it
  is.

### Probe record

Two live probes ran in scratch space against docker 29.1.3 and both changed a
confirmed claim:

1. **Changing `container_name` on a running service** → compose matched by the
   `com.docker.compose.service` label, printed `Recreated`, started the new name
   and left **zero orphans**. This disproved `c14`'s manual-reap premise
   (`c32`, `s22`).
2. **`container_name: ${X:-default}` interpolation** → with the key absent the
   default holds and a re-up prints **`Running`**, not `Recreated`; with the key
   set the rename happens with no orphan (`s28`). This proved the deferred
   override design viable — recorded in `c35` and parked as `v5`.

## Open parks

- [follow_up] The <ROLE>`_CONTAINER_NAME` override + env-resolved constants refactor: probed viable (absent key holds the default, a re-up prints 'Running', so an existing box could upgrade with zero downtime and no model reload) but deliberately deferred out of PR2. Picking it up also answers q3 and removes the hardcoding that made #111 possible

## Resolved vagueness

- [unknown_blocking] Whether renaming `container_name` is worth its cost at all: the names are invisible to every documented CLI path (lobes logs, status and doctor all resolve them internally) and only surface to an operator running raw docker ps — so the benefit may be identity tidiness against a cost of container recreation plus lock drift on three live boxes — resolved: YES, rename. Depth escalates from (b) to (c): the 18 `container_name` entries, the 14 `_compose.py` constants, mg-logwrap.sh and the /logs/model-gear mount path are all IN. Depth (d) is still NOT taken — the model-gear PyPI alias keeps being published.
- [unknown_nonblocking] Whether the fleet compose still needs `MODEL_GEAR_LOG_DIR` as an operator knob at all, or whether the rename is the moment to fold it into the profile/shape render path like every other knob — resolved: `MODEL_GEAR_LOG_DIR` survives as a read fallback. Verified it is NOT already a fallback — `_compose.py`:30 `LOG_DIR_ENV` = '`MODEL_GEAR_LOG_DIR`' is the sole key — so the operator's rule 'can survive as fallback for now; if it's already fallback, can remove' lands on the first clause: add `LOBES_LOG_DIR` as primary, keep reading the old key. Folding it into the profile render path is NOT done here.
- [unknown_blocking] Whether the deployed docker compose on the Thor and Orin supports the nested-default interpolation ${`LOBES_VERSION`:-${`MODEL_GEAR_VERSION`:-}} — verified only against docker 29.1.3 on the Spark; if an older compose on a Jetson does not, c28's fix does not work there and the compose-side key must stay legacy — resolved: Use `LOBES_VERSION` and ERROR if missing — fail closed, no nested-default interpolation. The compose build ARG becomes `LOBES_VERSION` and an absent value is a named startup error rather than 'pip install lobes-cli==' resolving to whatever is latest. This sidesteps the unverified ${A:-${B:-}} nesting on the Jetsons' docker compose entirely; lobes init's merge-only append is what puts `LOBES_VERSION` into an upgraded .env.
- [unknown_nonblocking] Whether any out-of-repo consumer (colleague, daria, reachy-mini-cli, eidetic-cli, culture mesh tooling) greps lobes container names or reads `MODEL_GEAR_`\* — the challenge pass examined this repo only and could not read sibling repos — resolved: Examined and CLEAN across the mesh. Scanned every local sibling repo plus ~/git on the Thor and the Orin over ssh: the Thor has zero legacy-name hits outside lobes-cli itself, and every Orin hit is a lobes-cli worktree (agent-qodo), i.e. this repo. Only two real cross-repo surfaces exist, both local and both minor: edge-ai-lab/tests/`test_arm_run.py`:498 writes `MODEL_GEAR_VERSION`=1 into a test .env (a genuine coupling, and the read-both-keys fallback keeps it working), and eidetic-cli/eidetic/memory/embed.py:29 names the embed/rerank containers in a COMMENT only.
