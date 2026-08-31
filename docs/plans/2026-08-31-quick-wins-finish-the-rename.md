# Build Plan — quick-wins + finish the rename

slug: `quick-wins-finish-the-rename` · status: `exported` · from frame: `quick-wins-finish-the-rename`

> lobes ships the #239 quick-wins batch and finishes the gear-to-lobes rename

## Tasks

### t1 — \#158 — drop the `MULTIMODAL_BASE_URL` non-empty default

- instruction: Edit lobes/templates/fleet/docker-compose.yml:1803 and lobes/templates/fleet/env.example:369 ONLY. Before flipping the default, check whether the shape renderer writes `MULTIMODAL_BASE_URL` on machine-as-brain; if it does not, add it there in the same commit or a senses-hosting box loses its wiring.
- acceptance:
  - fleet/docker-compose.yml:1803 reads ${`MULTIMODAL_BASE_URL`:-} and fleet/env.example:369 no longer presets a URL
  - a fleet deployment that never ran vllm-multimodal reports senses loaded:false in GET /capabilities
  - a deployment that DOES host senses still reports loaded:true — the shape renderer writes the key explicitly

### t2 — \#120 — move the three multimodal-family lanes off the dead `VLLM_ATTENTION_BACKEND` env

- instruction: Edit lobes/templates/fleet/docker-compose.yml lines 1032, 1189 and 1297 ONLY. The same file's :840 comment and env.example:318/:500 already say this env is dead on the pinned nightly — follow the cortex lane's --attention-backend flag form. Rebase on t1; do not re-edit t1's lines.
- depends on: t1
- acceptance:
  - fleet/docker-compose.yml:1032/1189/1297 no longer set `VLLM_ATTENTION_BACKEND`; each lane passes its backend via the flag the cortex lane uses
  - the Thor's validated `TRITON_ATTN` divergence for those lanes is preserved and asserted by a test

### t3 — \#111 — serve health-waits the container the deployment actually runs

- instruction: lobes/cli/`_commands`/serve.py:49 passes the default container=`_compose`.CONTAINER (the legacy single-model name). Resolve the container from the deployment shape instead. Use the constants AS THEY ARE SPELLED TODAY — the rename is PR2 and rebases on this.
- covers: c4, h3
- acceptance:
  - lobes serve --apply on a FLEET deployment waits on the fleet primary, not the legacy single-model name, and reports success
  - a unit test asserts the resolved container differs between fleet and --single scaffolds

### t4 — \#172 — conftest-level hardware-detection pin

- instruction: Add the autouse fixture to tests/conftest.py defaulting lobes.runtime.`_detect`.`detect_card` to UNKNOWN. Leave the per-file spark pins in tests/`test_init.py` and tests/`test_cli_logs.py` in place — they must still override it, and no existing assertion changes.
- acceptance:
  - tests/conftest.py carries an autouse fixture defaulting `detect_card` to UNKNOWN
  - the per-file spark pins in `test_init.py` and `test_cli_logs.py` still override it and their assertions are unchanged

### t5 — \#175 — lobes init discloses which profile file won

- instruction: lobes/cli/`_commands`/init.py — one disclosure line only. Precedence stays operator-wins; this is purely about saying which file won. The doctor check and the two-file diff from issue #175 are NOT in scope.
- acceptance:
  - init names an operator profile that shadows a built-in, quoting the path
  - a built-in-only resolution prints no shadow line

### t6 — \#99 — a named opt-in re-pins the deployed version key

- instruction: lobes/cli/`_commands`/doctor.py — the skew is already detected by `_version_skew_check`; add only the WRITE, behind its own named flag. Doctor's never-rewrite-an-existing-.env-line convention is load-bearing (#174, #191): without the flag, a .env with an existing version line must be byte-identical after --fix --apply.
- covers: c5, h4
- acceptance:
  - the re-pin is reachable only behind its own named flag
  - a .env with an existing version line is byte-identical after doctor --fix --apply WITHOUT that flag

### t7 — \#171 — annotate the Orin KV figure with its boot order

- instruction: docs/orin-profiles.md:33 and docs/machine-profiles.md ONLY. Cite the 2026-08-04 measurement for the gears-first 480,431/3.67x figure; do not restate it from memory. Do not touch lobes/profiles/builtin/orin.toml.
- covers: c3, h10
- acceptance:
  - docs/orin-profiles.md:33 names the senses-first boot order and adds the gears-first 480,431/3.67x counterpart, cited to the 2026-08-04 measurement
  - docs/machine-profiles.md states boot order changes the served KV budget, not just OOM risk
  - lobes/profiles/builtin/orin.toml is unchanged

### t8 — PR1 assembly — verify each of the seven at its cited line, bump, open

- instruction: Reproduce each of the seven at its cited line BEFORE writing its fix; drop any already fixed rather than reporting it fixed. Bump with .claude/skills/version-bump/scripts/bump.py (pipe a changelog JSON — it blocks on stdin — then commit the uv.lock re-pin). Open via the cicd skill; reply to and resolve every review thread.
- depends on: t1, t2, t3, t4, t5, t6, t7
- covers: c1, h1, c2, h2, c20, h21
- acceptance:
  - each of the seven defects is reproduced at its cited file and line before its fix is written; any already fixed is dropped from the batch rather than reported fixed
  - the version is bumped via .claude/skills/version-bump/scripts/bump.py and version-check passes
  - every review thread is replied to and resolved before merge

### t9 — PR2/templates — `container_name` defaults, the `MG_` log vars, the mount path, and the version build ARG

- instruction: lobes/templates/ ONLY. Rename every `container_name`: default, `MG_LOG_DIR`/`MG_LOG_NAME` and the /logs/<name> mount point, and make the build ARG `LOBES_VERSION`. NO nested-default interpolation — the fail-closed decision (frame v3) means an absent `LOBES_VERSION` is an error, not a fallback.
- depends on: t8
- covers: c29, h29
- acceptance:
  - every `container_name`: in the templates spells the new default
  - `MG_LOG_DIR`/`MG_LOG_NAME` and the /logs/<name> mount point are renamed in the same commit; existing host-side logs are not migrated
  - the build ARG is `LOBES_VERSION` with no nested-default interpolation

### t10 — PR2/init.py — merge-only append of the new version key

- instruction: lobes/cli/`_commands`/init.py ONLY. Append `LOBES_VERSION` on a pre-existing deployment; never rewrite an existing line (#191). This is what supplies the key that t12's fail-closed build requires.
- depends on: t8
- covers: c8, h5
- acceptance:
  - lobes init --apply on a pre-existing deployment appends `LOBES_VERSION` and rewrites no existing line
  - a box whose .env carries only the legacy key still resolves it everywhere Python reads it

### t11 — PR2/`_compose.py` — container constants, the log-dir two-key read, and the wrapper template entry

- instruction: lobes/runtime/`_compose.py` ONLY. Rename CONTAINER + the 13 `FLEET_`\* constants, make `LOG_DIR_ENV` a two-key read (`LOBES_LOG_DIR` then `MODEL_GEAR_LOG_DIR`), and register the renamed wrapper in `FLEET_TEMPLATES` in the same commit. Mirror the `LOBES_DIR`/`MODEL_GEAR_DIR` pattern already at :324-336 — read it as the reference, do not rewrite it.
- depends on: t8
- covers: c18, h9
- acceptance:
  - CONTAINER and the 13 `FLEET_`\* constants spell the new lobes-\* names
  - `LOG_DIR_ENV` resolves `LOBES_LOG_DIR` first and `MODEL_GEAR_LOG_DIR` second; a test proves `LOBES_` wins when both are set AND the legacy key alone still resolves
  - `FLEET_TEMPLATES` registers the renamed wrapper in the same commit as the compose reference

### t12 — PR2/Dockerfiles — the version pin fails closed

- instruction: The three Dockerfiles ONLY. An absent `LOBES_VERSION` must produce a NAMED error naming 'lobes init --apply' as the fix — never 'pip install lobes-cli==' resolving to latest. Verify by actually building the gateway image against a legacy-key-only .env.
- depends on: t9
- covers: c28, h24
- acceptance:
  - Dockerfile.gateway, .realtime and .chatterbox pin lobes-cli==${`LOBES_VERSION`}
  - building the gateway image against a .env carrying ONLY the legacy version key fails with a NAMED error telling the operator to run lobes init --apply — it never installs an unpinned wheel

### t13 — PR2/explain — refresh the taught content

- instruction: lobes/explain/catalog.py lines 137, 204, 280-282, 315, 547, 1505. Do NOT touch line 1525 — the ('model-gear',) alias is the deprecated dist/repo name and must keep resolving.
- depends on: t11
- covers: c36, h28
- acceptance:
  - lobes explain and lobes learn name no legacy container at catalog.py:137/204/280-282/315 and no legacy version key at 547/1505
  - the ('model-gear',) back-compat alias at catalog.py:1525 still resolves

### t14 — PR2/scripts — move the seven acceptance scripts onto the new names

- instruction: The seven scripts under scripts/. They are NOT covered by the test suite, so grepping is not enough — run scripts/live-check.sh against a real box before calling this done.
- depends on: t11
- acceptance:
  - accept-by-proxy.sh, accept-shape.sh, live-check.sh, `probe_reranker_calibration.py`, spec-arms.py, spike-preflight.sh and validate-tiers.sh reference no legacy name
  - scripts/live-check.sh runs green against a real box after the rename — the scripts are exercised, not only grepped

### t15 — PR2/prose — docs and CLAUDE.md, with the frozen record left alone

- instruction: docs/\*.md and CLAUDE.md prose. NEVER a repo-wide sed: docs/evidence and .devague are frozen verbatim records, and CHANGELOG.md's historical entries describe releases that really were named that. Verify with: git diff --stat main -- docs/evidence .devague
- depends on: t13
- covers: c9, h11, c30, h30
- acceptance:
  - docs/\*.md and CLAUDE.md carry the new names
  - git diff --stat main -- docs/evidence .devague prints NOTHING
  - CHANGELOG.md gains only its new release entry; no historical entry is edited

### t16 — PR2/lock — copy forward rather than invalidate

- instruction: lobes/runtime/`_lock.py`. Copy forward, never rename in place: emit the new \[files\] names alongside the old, and leave a renamed scaffold's old file on disk. Test against a FIXTURE lock carrying the OLD names — deployments/ holds no real one.
- depends on: t11
- covers: c26, h19
- acceptance:
  - a deployment.lock.toml written before PR2 still restores after it, proven against a FIXTURE lock carrying the OLD file names
  - the renamed scaffold leaves the old file on disk rather than deleting it

### t17 — PR2/guards — the surfaces that must NOT move

- instruction: Guard tests only. tests/fixtures/`upgrade_compat`/\*\*/docker-compose.yml must stay byte-identical — if `test_upgrade_compat` fails, the production code is wrong, not the fixture. Also assert the publish workflow still builds packaging/model-gear/ and that no code path writes a deployment directory.
- depends on: t11
- covers: c12, h7, c10, h12, c16, h13, c24, h17
- acceptance:
  - tests/fixtures/`upgrade_compat`/{single,fleet}/docker-compose.yml are byte-identical and `test_upgrade_compat` passes
  - the publish workflow still builds the deprecated PyPI alias and pip install model-gear still resolves to lobes-cli
  - no code path added by PR2 writes or migrates a deployment directory; ~/.model-gear stays a valid --compose-dir target

### t18 — PR2 assembly — re-scan, evidence, bump, open

- instruction: Branch off PR1's HEAD, not main. Re-run the zero-locks scan (ls ~/.lobes/deployment.lock.toml on each box) and paste the result into the PR body; if a lock now exists, withdraw the cheapest-now argument and exercise t16's copy-forward for real.
- depends on: t12, t14, t15, t16, t17
- covers: c11, h6, c17, h8, c22, h22
- acceptance:
  - the zero-captured-locks scan is RE-RUN at PR2 time and its result pasted into the PR body; if a lock has appeared, the cheapest-now argument is withdrawn and the copy-forward path is exercised for real
  - PR2 branches off PR1's head, not main, and the serve.py fix survives the constant rename intact
  - the version is bumped and version-check passes

### t19 — Delivery verification — measure the success signals, do not assert them

- instruction: MEASURE, do not assert. Run each check and paste its output: gh issue list for the seven; git diff --stat main -- docs/evidence .devague; the c23 grep; uv run pytest -n auto -q. A number you did not run is not a success signal.
- depends on: t18
- covers: c19, h20, c21, h14, c23, h15, c27, h23
- acceptance:
  - all 7 of the #239 items are closed by PR1
  - 0 files changed under docs/evidence and .devague across both PRs (git diff --stat)
  - 0 occurrences of model-gear outside packaging/model-gear, tests/fixtures/`upgrade_compat`, docs/evidence, .devague and CHANGELOG.md after PR2 — the c23 grep is RUN and its output pasted into the PR body
  - 3 of 3 env key pairs resolve from the legacy key alone in tests
  - the full suite is green with 0 failures including `test_upgrade_compat`
  - lobes serve on a fleet deployment waits on a container that exists and reports success

## Risks

- [unknown_nonblocking] Both container-name probes ran on docker 29.1.3 on the Spark, not on the Jetsons. They prove compose's service-label matching, which is not version-fragile, but that is inference — a Jetson with an older compose could behave differently on the one restart PR2 causes
- [unknown_nonblocking] Each renamed lane RESTARTS once on the three live boxes; for a 27B NVFP4 cortex that is a model reload of minutes. The override design that would avoid it is deliberately deferred (frame park v5), so PR2's rollout must be scheduled when that downtime is acceptable
- [follow_up] edge-ai-lab/tests/`test_arm_run.py`:498 writes the legacy version key into a test .env. The read-both-keys fallback keeps it working, but that repo is out of tree and nothing in this plan updates it
