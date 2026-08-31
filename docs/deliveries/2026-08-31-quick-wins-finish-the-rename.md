# Delivery Summary — quick-wins + finish the rename

plan: `quick-wins-finish-the-rename` · run: `partial` · date: `2026-08-31`
baseline: `devague summary skeleton`

## Intent

Ship the seven small fixes batched in issue #239 as PR1, then finish the
gear→lobes rename at depth (c) as PR2. This artifact covers a **partial** run:
PR1 (`t1`–`t8`) is code-complete and awaiting its PR; PR2 (`t9`–`t19`) has
not started.

## Planned Work

Quoted verbatim from the `devague summary` skeleton:

- `t1` — #158 — drop the `MULTIMODAL_BASE_URL` non-empty default
- `t2` — #120 — move the three multimodal-family lanes off the dead `VLLM_ATTENTION_BACKEND` env
- `t3` — #111 — serve health-waits the container the deployment actually runs
- `t4` — #172 — conftest-level hardware-detection pin
- `t5` — #175 — lobes init discloses which profile file won
- `t6` — #99 — a named opt-in re-pins the deployed version key
- `t7` — #171 — annotate the Orin KV figure with its boot order
- `t8` — PR1 assembly — verify each of the seven at its cited line, bump, open
- `t9` — PR2/templates — `container_name` defaults, the `MG_` log vars, the mount path, and the version build ARG
- `t10` — PR2/init.py — merge-only append of the new version key
- `t11` — PR2/`_compose.py` — container constants, the log-dir two-key read, and the wrapper template entry
- `t12` — PR2/Dockerfiles — the version pin fails closed
- `t13` — PR2/explain — refresh the taught content
- `t14` — PR2/scripts — move the seven acceptance scripts onto the new names
- `t15` — PR2/prose — docs and CLAUDE.md, with the frozen record left alone
- `t16` — PR2/lock — copy forward rather than invalidate
- `t17` — PR2/guards — the surfaces that must NOT move
- `t18` — PR2 assembly — re-scan, evidence, bump, open
- `t19` — Delivery verification — measure the success signals, do not assert them

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered (different fix) | The default was **restored**, not dropped. `loaded` now derives from feasibility (`lobes/roles.py`). See Drift. |
| `t2` | delivered | Three lanes moved to `--attention-config`; the dead env is gone; the stale #109 guard retired. |
| `t3` | delivered | `serve.py` resolves the container from the deployment shape (colleague-authored). |
| `t4` | delivered | Autouse UNKNOWN card pin in `tests/conftest.py` (colleague-authored). |
| `t5` | delivered | `_operator_shadow_line` in `init.py` (colleague-authored). |
| `t6` | delivered | `lobes doctor --repin-version`, behind its own flag. |
| `t7` | delivered | Boot-order annotation + gears-first counterpart (colleague-authored). |
| `t8` | partial | Seven verified at their surfaces; version bumped to 0.73.0; rubric green. **PR not yet opened.** |
| `t9`–`t19` | blocked | PR2 not started; gated on PR1 merging. |

## Mid-work Decisions

No `devague deviate` records exist for this run (`devague deviate --list` →
"no deviations recorded yet"). These decisions are captured here directly.

- **`t1`: the issue's proposed fix was wrong and was abandoned.** #158 said to
  empty the `MULTIMODAL_BASE_URL` compose default "to match every sibling".
  The siblings default empty because they are **opt-in** roles with activation
  tables in `shape_render.py`; `senses` is an **always-on core** role whose
  only wiring *is* that default (`render.py::profile_env` leaves
  `${VAR:-default}` in effect for anything a profile is silent on). Emptying
  it breaks every senses-hosting box.
- **A second approach was tried and reverted.** Adding a `CORE_ACTIVATION_ENV`
  for senses works semantically but breaks the pinned
  `test_machine_as_brain_env_equals_profile_env` invariant — 6 failures → 12.
- **`t2`: a recorded deviation in the code turned out to be stale.** The compose
  carried "DELIBERATELY LEFT AS-IS … gated on a GB10 live-verification tracked
  in #109", with a test class guarding it byte-for-byte. **#109 is closed**, with
  the explicit routing "env dead on GB10 → t3 deletes as planned".
- **`t2`: deleting the env outright was rejected.** All three card profiles
  declare `senses.attention_backend`, so `profile_env` renders
  `MULTIMODAL_ATTENTION_BACKEND` into every `.env`; deleting the consumer would
  orphan a live key (#204's complaint). The knob migrated instead.
- **Workforce concurrency was tuned down mid-run.** Three parallel colleague
  seats on one `cortex` lane pushed model turns to 393–642s against a 300s
  timeout; 2 of 3 stalled. Reduced to 1–2.
- **`t1` and `t2` were taken in-house after their colleague flights stalled.**
  Both needed repo archaeology (a pinned invariant; a closed gate) rather than
  a code edit.

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|-----------------------|----------------|
| `t1` | The task's acceptance criteria described emptying the compose default. That change was made, proven wrong, and reverted; the delivered fix is a one-line change in `lobes/roles.py` instead. The **user-visible outcome** the criteria aimed at (senses not reporting `loaded: true` on a box that does not host it) is delivered. | acceptable |
| `t2` | The plan assumed the migration was unblocked. It was gated by an in-code deviation citing #109; that gate had closed, so the guard and its comment were retired as part of the task. | acceptable |
| `t8` | Assembly is complete except for opening the PR, which is the next step. | acceptable |
| `t9`–`t19` | PR2 not started — the plan sequences it behind PR1's merge (`t17`/`c17`). | acceptable |
| — | `lobes measure` now skips a **proxied** role, because `roles_measure.py` branches on `loaded` in four places. No plan task anticipated this. Filed as **#240**. | needs-follow-up |

## Evidence

- tests: full suite `uv run pytest -n auto -q` — **4419 passed, 15 skipped**
- tests: `tests/test_cli_capabilities.py::test_senses_is_not_loaded_on_a_box_that_declares_it_infeasible` — pass; **verified falsifiable** (fails on the pre-fix line)
- tests: `tests/test_doctor_repin_version.py` — 7 passed, incl. `test_the_heal_lane_still_never_rewrites_an_existing_line`
- lint: `uv run black/isort/flake8` — clean; `markdownlint-cli2` — 0 errors
- rubric: `uv run afi cli doctor . --strict` — **exit 0**
- commits: `0dedf16..50cb4a6` (21 commits)
- issues: #158 #120 #111 #172 #175 #99 #171 (fixed) · #109 (closed gate, cited) · #204 (avoided) · **#240** (filed) · colleague#473 (practice record)

## Delivery Claims

| Claim | Confidence | Evidence |
|-------|------------|----------|
| A role declared infeasible no longer reports `loaded: true` | high | test `test_senses_is_not_loaded_on_a_box_that_declares_it_infeasible` · `lobes/roles.py` |
| `lobes serve` waits on the container a fleet deployment runs | high | `lobes/cli/_commands/serve.py` · `tests/test_cli_runtime.py` |
| The three Gemma lanes select their backend via `--attention-config` | high | `lobes/templates/fleet/docker-compose.yml` · `tests/test_fleet_per_machine_knobs.py` |
| `doctor --repin-version` writes the pin; `--fix --apply` still never rewrites | high | `tests/test_doctor_repin_version.py` (7 tests) |
| The suite no longer inherits the runner's GPU card | high | `tests/conftest.py` autouse pin |
| `lobes init` names an operator profile that shadows a built-in | high | `lobes/cli/_commands/init.py::_operator_shadow_line` |
| The Orin KV figure carries its boot order | high | `docs/orin-profiles.md` · `docs/machine-profiles.md` |
| The rename (PR2) is delivered | unverified | not started — not claimed done |
| The three lanes serve correctly on real hardware after the flag migration | unverified | no live boot on Thor/Orin/Spark since the change |

## Remaining Work / Follow-up

- `t8` — open PR1 via the `cicd` skill; run **both** review passes (colleague + Qodo) before merge.
- `t9`–`t19` — PR2, the depth-(c) rename. Branch off PR1's head, not main. The 18-pair container name table is pinned; `FLEET_FALLBACK` is **dead** (no template declares it) and should be deleted rather than renamed.
- **#240** — `lobes measure` skips a proxied role; needs `proxied` on `RoleInfo`. `assess.py:82` carries the same predicate.
- **Live verification** — no box has booted the migrated attention flag. Required before claiming the `t2` lanes serve.
- **#239's framing** should be corrected: two of seven items had premises contradicted by the code. "Known file and line" did not imply "known fix".
