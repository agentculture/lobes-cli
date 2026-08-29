# Delivery Summary — deployment lock per box

plan: `deployment-lock-per-box` · run: `partial` · date: `2026-08-29`
baseline: `devague summary skeleton`

## Intent

Turn issue #214's practice — commit each box's rendered compose/Dockerfiles as
a deployment lock, keep secrets out of git — into a shipped mechanism. Four
operator decisions taken during `/think` reshaped it from a per-box backup into
a **published catalog of deployment variations**, keyed by machine type, whose
secrets are generated or file-supplied rather than ever retyped. Twelve tasks
were fanned out across three dependency waves by `/assign-to-workforce`.

**All twelve tasks merged and the full suite is green — but this run is
`partial`, not `complete`,** because three tasks delivered less than their
acceptance criteria promised and the feature cannot yet be exercised end to end
(there is no capture verb). The task-by-task accounting below says exactly
where.

## Planned Work

Quoted verbatim from the `devague summary` skeleton:

- `t1` — Positional gitignore rule + goldens negation
- `t2` — Variation identity resolver (machine type, never hostname)
- `t3` — CI secret gate over every committed deployment artifact
- `t4` — Secret env family: second `env_file` entry across fleet services
- `t5` — Wheel-exclusion test for the deployments tree
- `t6` — Lock writer: allowlist from `profile_env` into deployment.lock.toml
- `t7` — lobes init --from-lock: verbatim materialise, bypass resolution, guard machine type
- `t8` — doctor `lock_drift` + switch staleness warning
- `t9` — Variation catalog: `deployments/<id>/` layout + info file contract
- `t10` — Buildability guard for a variation's pinned `MODEL_GEAR_VERSION`
- `t11` — Secret rotation and revocation path
- `t12` — Docs + framing: audience, before/after state, why it matters, success signals

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | `.gitignore` positional rule (`*.env` ignored, `.env.*` allowed) plus `!tests/goldens/**/*.env`; 9 tests in `tests/test_gitignore_convention.py` built against a scratch repo seeded from the real shipped file, including an adversarial test that strips the negation and asserts the goldens become unstageable |
| `t2` | delivered | `lobes/variation.py` — `resolve_variation_id(DetectedCard) -> str` and `UNKNOWN_VARIATION`, a thin seam over existing detection rather than a second probe path; 5 injected-fact tests |
| `t3` | delivered | `scripts/scan_deployment_secrets.py` (stdlib) plus a required `secrets-scan` CI job; 13 fixture tests. Extended during integration with a second scan root over `tests/fixtures` |
| `t4` | delivered | `.secrets.env` as a `required: false` second `env_file` entry across 12 fleet services and 2 audio services; `gateway` and `llamacpp-primary` correctly left alone; 9 parse-level tests |
| `t5` | delivered | `tests/test_wheel_excludes_deployments.py` — an always-on static assertion on `packages = ["lobes"]` plus a build layer that builds the wheel twice (normal, and with `packages` widened) to prove the exclusion is load-bearing |
| `t6` | delivered | `lobes/runtime/_lock.py` — `deployment.lock.toml`, allowlist derived from `profile_env`, hand-rolled TOML writer, 60 tests including one that proves a denylist implementation would fail |
| `t7` | delivered | `lobes init --from-lock` + `--allow-variation-mismatch`; 38 tests. Criterion 4 proven by patching all seven resolution entry points *and* `write_scaffold` to raise |
| `t8` | partial | `lock_drift` in doctor and a `switch` staleness warning, 10 tests — but the mechanism it had to adopt (tracking `.env`'s digest) is incompatible with `t7`'s restore path; see `d6` |
| `t9` | partial | `deployments/` catalog, `VARIATION.template.md`, `lobes/variation_catalog.py` validator, 28 tests, 2 fixture variations — but **zero real variations**, because capturing one needs physical hardware |
| `t10` | partial | `lobes/runtime/_buildability.py`, 27 hermetic tests — shipped with no caller; wired during integration as an **offline, warn-only** preflight, so its "fails early" criterion is met only in the warning sense |
| `t11` | delivered | `docs/secret-rotation.md` — the full O(machines) key-copy inventory across ten role prefixes, a 5-step drill, and the `git rm` / history-rewrite statement. Explicitly flagged as reviewed-against-source, not run |
| `t12` | delivered | `docs/deployment-lock.md`, `lobes explain lock` (+ updates to `explain init`/`doctor`/`switch`), a `CLAUDE.md` section, 9 doc-vs-tree cross-check tests and 8 explain tests |

## Mid-work Decisions

Six deviations were recorded during the run and **all six were approved by the
plan owner** on 2026-08-29, after the summary was first written. They are the
recorded ground truth this section and the drift table below quote.

- `d1` (approved, `acceptable`) — the lock's key set is an allowlist **minus
  two exclusion rules**, not the pure allowlist the spec claims: `COMPOSE_PROFILES`
  (which `profile_env` does emit, and which `c4` names as operator-typed) and
  any `_URL`-suffixed key (the shape layer's `*_BASE_URL` / `PRIMARY_URL`
  wiring, hand-retargetable at a peer box). Both rules only ever *remove*
  keys, so the failure mode `c25` was written against remains impossible.
- `d2` (approved, `needs-follow-up`) — a lock-restored **fresh** box is not
  servable: the fleet compose bind-mounts `mg-logwrap.sh` and the tool-parser
  plugin, which are packaged scaffold files rather than compose/Dockerfiles.
- `d3` (approved, `needs-follow-up`) — `t10`'s guard shipped with no caller,
  because `t10` was forbidden from editing the only natural wiring points
  (`init.py`, `doctor.py`), both owned by sibling tasks.
- `d4` (approved, `acceptable`) — **spec claim `c33` is wrong.** `lobes switch`
  writes only legacy `VLLM_*` keys, which never intersect `lock_keys()`. The
  claim's conclusion survives (a first-class verb does make a lock stale) but
  its stated mechanism does not.
- `d5` (approved, `needs-follow-up`) — **there is no capture verb.**
- `d6` (approved, `risky`) — `d4`'s staleness mechanism and `t7`'s restore path
  cannot both use one lock.
- Not covered by any record: the integration pass deliberately made the
  buildability preflight **warn-only rather than raising**, because
  `lobes_version` is optional in the lock schema — absence means "not
  recorded", never "broken", and raising on it rejected every fixture
  variation (20 test failures caught this).

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|-----------------------|----------------|
| `t6` (`d1`) | the delivered lock is an allowlist minus two exclusion rules, not the pure allowlist `c25` and `h20` describe | acceptable |
| `t7` (`d2`) | a restored fresh box lacks the two bind-mounted scaffold files, so `c1`/`h1`'s "byte-identical restore" does not yield a servable box | needs-follow-up |
| `t10` (`d3`) | shipped with no caller; wired later as offline warn-only, so `h28`'s "reported unbuildable" is a warning, never a refusal | needs-follow-up |
| `t8` (`d4`) | `c33`'s mechanism is factually wrong; staleness had to be caught by an `.env` digest instead | acceptable |
| `t6`/`t7` (`d5`) | no CLI path captures a lock, so the practice cannot be exercised end to end — `c14`/`h10`'s adoption promise is unreachable today | needs-follow-up |
| `t7`/`t8` (`d6`) | a drift-capable lock is not restorable and a restorable lock cannot detect `.env` drift; nothing reconciles them | risky |
| `t9` | zero real variations shipped — `h10` ("materialisable on a machine that never served it") is tested only against fixtures, since capture needs physical hardware | needs-follow-up |

## Evidence

- tests: full suite `uv run pytest -n auto -q` — **4194 passed, 15 skipped**
  (the 15 are pre-existing live-gateway/optional-dependency gates, unchanged
  by this run)
- tests: `tests/test_deployment_lock.py` (60), `tests/test_init_from_lock.py`
  (38), `tests/test_variation_catalog.py` (28), `tests/test_buildability_guard.py`
  (27), `tests/test_scan_deployment_secrets.py` (13), `tests/test_lock_drift.py`
  (10), `tests/test_gitignore_convention.py` (9), `tests/test_fleet_secrets_env_file.py`
  (9), `tests/test_deployment_lock_doc.py` (9), `tests/test_variation.py` (5),
  `tests/test_wheel_excludes_deployments.py` (4) — all pass
- lint: `uv run black --check lobes tests`, `uv run isort --check-only lobes tests`,
  `uv run flake8 lobes tests` — clean
- security: `uv run bandit -c pyproject.toml -r lobes` — 0 low / 0 medium / 0 high
- rubric: `uv run afi cli doctor . --strict` — exit 0
- scanner: `uv run python scripts/scan_deployment_secrets.py --root .` — clean;
  `--root tests/fixtures` — clean (8 files scanned)
- markdown: `markdownlint-cli2` — 0 errors on every page added
- commits: `eb0174a..656f167`
- issues: #214 (the practice), #204 (the per-box override gap this routes
  around, still open), #108 (the declared-vs-validated rule)

## Delivery Claims

| Claim | Confidence | Evidence |
|-------|------------|----------|
| The gitignore convention is positional and the goldens negation is load-bearing | high | `tests/test_gitignore_convention.py::test_negation_removal_breaks_goldens_staging` |
| The lock is secret-free by construction, not by redaction | high | `tests/test_deployment_lock.py::test_injecting_any_new_key_never_changes_the_lock` |
| `--from-lock` genuinely bypasses profile/shape resolution | high | `tests/test_init_from_lock.py` — resolution entry points patched to raise; restore still passes |
| A restore leaves an existing `.env` byte-identical | high | `tests/test_init_from_lock.py::test_restore_leaves_an_existing_env_byte_identical` |
| The CI secret gate fails on a planted token and passes clean | high | `tests/test_scan_deployment_secrets.py` · `.github/workflows/tests.yml` job `secrets-scan` |
| `lock_drift` names the specific differing files | high | `tests/test_lock_drift.py` |
| Committed deployment artifacts never ship in the wheel | high | `tests/test_wheel_excludes_deployments.py` |
| **A restored box serves the same model at the same knobs as before the restore** | unverified | no box has been restored and served; argued from byte-identical files + merge-only `.env`, never measured |
| **A third party can adopt a variation they never ran** | unverified | zero real variations exist; tested against fixtures only |
| **The buildability check fails early rather than inside a docker build** | low | it warns, never refuses; the raising path needs an opt-in index query that is not wired (`d3`) |
| **The practice is usable end to end** | unverified | no capture verb exists (`d5`) — verified by absence of any `capture_lock`/`write_lock` caller under `lobes/cli/` |
| The secret-rotation drill is correct | unverified | reviewed against source, never run against a live fleet |

## Remaining Work / Follow-up

1. **Add a capture verb** (`d5`) — the highest-priority gap. Without it the
   library can restore but nothing can produce a lock, so every "re-capture the
   lock" remediation string in doctor, switch and the catalog README is
   currently an instruction to call a Python function.
2. **Reconcile `d6`** — decide whether the lock's `[files]` table tracks `.env`
   for drift detection or stays restorable, and make the two paths agree.
   Classified `risky` because the conflict is silent: each side has passing
   tests.
3. **Resolve `d2`** — decide whether a variation's `[files]` should name the
   bind-mounted scaffold files. The restore mechanism already supports it; no
   capture path records them and no test covers it.
4. **Capture a real variation** on a Spark, Thor or Orin — this is what turns
   the catalog from a contract into a catalog, and what would let the two
   `unverified` claims above be measured. Requires hardware.
5. **Wire an opt-in index query** so the buildability guard can refuse rather
   than warn (`d3`).
6. **Reconcile the spec with `d1`** — the deviation is approved, but the
   exported spec still states `c25` in its original pure-allowlist form.
   Amending the claim (or recording the exclusions alongside it) keeps the
   spec and the delivered mechanism from disagreeing.
7. Carried from the frame, still open: no trust model exists for adopting a
   third party's variation (the challenge pass recorded this surface as
   unexamined), and `lock_drift` can only verify the local variation.
