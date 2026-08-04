# Delivery Summary — unsloth QAT senses + first-class orin variation

plan: `unsloth-qat-senses-first-class-orin-variation` · run: `complete` · date: `2026-08-04`
revised: `2026-08-04` — the live boot the operator originally could not authorise was
subsequently approved and run; `t9`/`t11` moved blocked → delivered, and the four
`unverified` capability claims below were re-stated with evidence.
baseline: `devague summary skeleton`

## Intent

> senses serves unsloth/gemma-4-12B-it-qat-w4a16, live-validated on this Jetson AGX Orin as a first-class orin variation, while the thor/spark machine profiles stay intact so moving the setup to another architecture stays a profile pick, not a rework

This run executed that plan's 13 tasks via `/assign-to-workforce`: three wave-0
repo tasks fanned out to isolated worktree agents in parallel, two box-side
measurement tasks run by the main agent, then three sequential worktree agents
for the profile/shape/compose layers, then the live boot and probe matrix on the
physical board. **Twelve of thirteen tasks delivered; `t10` is partial** — its
cortex half is verified, its worker half is wired but gated on releasing 0.55.0.
See Drift and Remaining Work.

## Planned Work

Quoted verbatim from the `devague summary` skeleton:

- `t1` — Catalog entry + per-model doc for unsloth/gemma-4-12B-it-qat-w4a16
- `t2` — Orin card detection strategy (lobes/machines/orin.py)
- `t3` — BOX: benchmark the incumbent coolthor on the current engine (BEFORE any swap)
- `t4` — BOX: snapshot ~/.lobes before any mutation
- `t5` — Parameterize the senses lane --speculative-config (MTP off-switch)
- `t6` — Builtin orin.toml profile (hypothesis knobs, measured-pending)
- `t7` — Builtin orin-lobe shape + iowait persistence + goldens regen
- `t8` — csv-mode GPU access knob (runtime:nvidia survives re-render)
- `t9` — BOX: re-render from the branch checkout + boot + measure the budget
- `t10` — BOX: proxy wiring — worker to Thor (new) + cortex to Spark (preserved)
- `t11` — BOX: capability probe matrix + pressure check on the live checkpoint
- `t12` — Evidence transcript + docs + measured-value backfill
- `t13` — PR: version bump, full suite, untouched-proof, consumer coordination + follow-up issue

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | `SupportedModel` entry + `docs/gemma-4-12b-qat-w4a16.md` with every capability row marked pending-live-probe. Merged `d88aa76`. |
| `t2` | delivered | `lobes/machines/orin.py` (sm_87 CardStrategy, own `role_overrides`, deliberately NOT composing the sm_110 trait) **plus** hermetic test isolation for `test_init.py`/`test_cli_logs.py`. Merged `577dfe4`. |
| `t3` | delivered | Incumbent baseline on the current engine: TTFT, decode tok/s from `usage.completion_tokens`, MTP acceptance, KV pool — plus a capability baseline (t3b) that validated the probe harness. In the evidence transcript. |
| `t4` | delivered | `~/lobes-snapshots/2026-08-04-pre-unsloth-swap/` verified byte-identical vs live, with `RESTORE.md`. |
| `t5` | delivered | `MULTIMODAL_SPECULATIVE_CONFIG` with true flag omission on an empty value; byte-guards strengthened, zero golden changes. Merged `2a0e0e1`. |
| `t6` | delivered | `lobes/profiles/builtin/orin.toml` (MEASURED-PENDING budgets) + `loader` now derives orin's overrides from the machines registry. Merged `0f98da1`. |
| `t7` | delivered | `orin-lobe` shape (no local stt/tts) + `Profile.host_env` rendering the Tegra iowait threshold from the **card**. Merged `f752ed8`. |
| `t8` | delivered | `Profile.gpu_access` + generated `docker-compose.gpu.yml` pair using `deploy: !reset null`; compose **merge** verified against real `docker compose config` on this csv-mode board. Merged `fb49eaa`. |
| `t9` | delivered | Re-rendered and **booted live**. `0.45` / `262144` accepted first try, no trim: weights 9.07 GiB, KV 11.81 GiB = 609,266 tokens = 2.32×. csv-mode `runtime: nvidia` proven at container **create**. Three blockers hit en route (`d6` shadowing, `d7` classifier, `d8`… plus #177). |
| `t10` | partial | cortex→Spark **verified** (`proxied: true`) after the `--force` rewrite. worker→Thor **wired** in `.env` (keyless) but inactive: the gateway image installs the *published* wheel, so 0.55.0 is undeployable until this PR merges. (`d5`, `d8`) |
| `t11` | delivered | Full matrix run against the new checkpoint, controlled against the incumbent baseline: image PASS, **video PASS** (incumbent FAILS the same reversed-motion control), reasoning PASS, tools PASS, audio FAIL (#101, vLLM-side). |
| `t12` | delivered | Evidence transcript carries the incumbent baseline, capability baseline, streaming contract, pre-boot facts, and the live boot + probe results. Measured values **backfilled** into `orin.toml` and `orin-lobe.toml` (lockstep test enforced); `orin-lobe` marked validated. |
| `t13` | delivered | Version 0.55.0, suite 2866 passed, untouched-proof verified, **six** issues filed, PR #176 open with all CI green and both Qodo findings fixed and resolved. |

## Mid-work Decisions

- `d1` — `docs/orin-profiles.md`'s KV figure (18.86 GiB / 802,644 tok / 6.12x) is a
  **1.67x overstatement** of what the live engine holds at those exact knobs
  (11.29 GiB / 480,431 tok / 3.67x). Cause: `gpu_mem_util` is a fraction of the
  **entire device**, so co-resident engines are deducted one-for-one from KV;
  the documented figure came from a senses-**first** boot. Filed as #171.
- `d2` — `t1` used `role_hint='candidate'`, not `'multimodal'`:
  `test_exactly_one_gemma_multimodal_gear` pins that tier to exactly one entry, and
  the existing Gemma coder entry already set the `'candidate'` precedent.
- `d3` — `t2`'s correct detection **exposed 20 host-dependent test failures**. The
  suite's result depended on which physical machine ran it; CI passed only because
  CI is not an Orin. Scope was **extended** to fix it hermetically rather than
  routed around. Filed as #172.
- `d4` — worker→Thor proxy first deferred by the user's senses-first priority ruling,
  then **folded back in** at `t10` once the user confirmed the Thor is keyless
  (voiding plan risk `r3`).
- `d5` — worker→Thor is **not** "two env lines": this box's gateway is 0.45.0, which
  has no `worker` role at all and predates the 0.54.8 peer-proxy fix. It needs a
  gateway rebuild, which `t9`'s re-render would have delivered.
- `d6` — an **operator profile shadowed the new builtin**: `~/.lobes/profiles/orin.toml`
  wins by name, so `t9`'s apply would have been a silent **no-op**. Caught in the dry
  run (a missing `docker-compose.gpu.yml` line was the only tell). Archived on-box,
  not deleted. Filed as #175.
- `d7` — the live re-render was initially **denied by the environment's permission
  classifier**, blocking every box-side task. The operator subsequently approved it
  and the boot ran; `t9`/`t11`/`t12` completed.
- `d8` — worker→Thor cannot activate until 0.55.0 is **published**: the gateway image
  installs `lobes-cli==${MODEL_GEAR_VERSION}` from PyPI, so a gateway-side feature on
  a branch is undeployable. Structural, not a defect. The env wiring is already in
  place and needs no further config after release.
- Not covered by any deviation record: the checkpoint ships **without
  `vision_config.num_soft_tokens`** and crash-loops vLLM every ~70 s. Exactly one key
  of twenty differs from its sibling export. Patched locally (280) to boot; filed as
  #177 because the patch lives in the HF cache and evaporates on a clear.
- Not covered by any deviation record: `--force` was found to **silently destroy 12
  operator-typed `.env` keys**, including the cortex peer-proxy credential. Discovered
  by rehearsing the render against a copy rather than running it live. Filed as #174,
  and the at-risk lines were captured verbatim before any apply was attempted.
- Not covered by any deviation record: issue #101's documented audio signature
  ("~19 placeholder tokens") is **wrong on this build** — audio tokens scale with
  duration (~25 tok/s). Its *conclusion* still holds, proven instead by a
  discrimination control. Correction posted to #101.

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|-----------------------|----------------|
| `t3` (`d1`) | the documented before-state KV figure was a 1.67x overstatement; boot order, not knobs, sets the measured pool — frame claim `c20` amended to measured reality per `h16` | needs-follow-up |
| `t1` (`d2`) | `role_hint='candidate'` preserves a pinned invariant; per-box selection happens via the orin profile as the plan intended | acceptable |
| `t2` (`d3`) | scope extended to fix 20 host-dependent tests the change exposed; TDD gate held the branch back until green | needs-follow-up |
| `t10` (`d4`) | deferred then reinstated by user rulings; senses remained the critical path throughout | acceptable |
| `t10` (`d5`) | requires a gateway rebuild (0.45.0 → 0.55.0), not the env-only change the plan assumed | risky |
| `t9` (`d6`) | operator profile shadowed the builtin; would have rendered a silent no-op | risky |
| `t9`, `t11`, `t12` (`d7`) | live mutation initially denied by the permission classifier; subsequently approved and completed — measured budget and full probe matrix now recorded | acceptable |
| `t10` (`d8`) | worker half gated on publishing 0.55.0 (the gateway installs the released wheel), so it ships wired-but-inactive | needs-follow-up |
| `t9` (#177) | the checkpoint's own `config.json` is incomplete and crash-loops vLLM; required a local patch the plan did not anticipate | needs-follow-up |

## Evidence

- tests: `uv run pytest -n auto` — **2860 passed, 15 skipped** (at `dd7be02`)
- lint: `black --check lobes tests` — 224 files unchanged; `flake8` — clean; `isort --check-only` — clean
- security: `bandit -c pyproject.toml -r lobes` — *No issues identified*
- rubric: `uv run afi cli doctor . --strict` — PASS
- commits: `ada7e67..dd7be02` (merges `d88aa76`, `577dfe4`, `2a0e0e1`, `0f98da1`, `f752ed8`, `fb49eaa`)
- evidence transcript: `docs/evidence/2026-08-04-accept-senses-unsloth-orin.txt`
- issues filed: #171, #172, #173, #174, #175 · correction on #101

## Delivery Claims

| Claim | Confidence | Evidence |
|-------|------------|----------|
| the unsloth checkpoint is a first-class catalog entry with an honest, unvalidated capability table | high | commit `d88aa76` · file `docs/gemma-4-12b-qat-w4a16.md` |
| this box detects as `orin` without `--profile` | high | live `lobes init` dry run reports `detected card='orin', compute_capability='sm_87'` |
| the test suite is now hermetic w.r.t. host hardware | high | commit `577dfe4` · parametrized test proving the injected card drives the outcome |
| the senses MTP draft can be switched off entirely | high | commit `2a0e0e1` · `tests/test_senses_speculative_config.py` |
| `orin-lobe` renders the orin senses budget, not thor's | high | golden `tests/goldens/shapes/orin-lobe__orin.env` (`0.45` / `262144`) · lockstep test |
| the Tegra iowait shed is fixed for **any** orin render, including bare `machine-as-brain` | medium | commit `f752ed8` · `[host_env]` on the card profile — rendered, not booted |
| a csv-mode board renders working GPU access with no hand editing | medium | real `docker compose config` exit 0, `runtime: nvidia → 12`, `driver: nvidia → 0` — compose **merge** proven, container create not |
| thor/spark profiles and all pre-existing goldens are byte-identical | high | `git diff main` over those paths is empty; golden diff is additions-only |
| the incumbent baseline is captured and unrecoverable-by-then metrics are preserved | high | `docs/evidence/2026-08-04-accept-senses-unsloth-orin.txt` |
| senses serves the unsloth checkpoint on this box | high | live `/capabilities`: `ready=True, ctx=262144`; evidence transcript t9 |
| the 0.45 / 262144 budget holds on real hardware | high | boot log: KV 11.81 GiB / 609,266 tokens, accepted first try |
| the new checkpoint understands VIDEO (the incumbent does not) | high | reversed-motion control: `LEFT-TO-RIGHT` vs `RIGHT-TO-LEFT`; incumbent answered `RIGHT-TO-LEFT` to both |
| image, reasoning and tool calling work on the new checkpoint | high | probe matrix, t11 section of the evidence transcript |
| the new checkpoint is faster than the incumbent | high | decode +8% medium, +79% long, measured from `usage.completion_tokens` |
| audio is usable on `senses` | **disproven** | discrimination control fails 0/3; model denies hearing audio it carries — #101 |
| **worker is reachable from this box via the Thor** | unverified | wired and peer-verified, but blocked on releasing 0.55.0 (`d8`) — NOT claimed done |

## Remaining Work / Follow-up

- `t10` (the only incomplete task) — tracked as **#178**: merge #176 → PyPI publishes
  0.55.0 → rebuild the gateway → `model=worker` answers with `X-Lobes-Proxied-By`. The
  `.env` wiring is already in place and needs no further configuration. #178 also
  carries the underlying gap: a gateway-side change cannot be deployed — and therefore
  cannot be live-validated — from a branch, because the image installs the published
  wheel.
- **#177 is the operational risk to watch**: the `num_soft_tokens` patch lives in this
  box's HF cache. A cache clear, re-pull, or fresh machine reintroduces the crash-loop.
  Needs an upstream fix or a pre-boot guard in `lobes doctor`.
- #171 — correct `docs/orin-profiles.md`'s KV figure and strengthen the boot-ordering caveat.
- #172 — the host-dependent-test class is fixed for two files; consider a conftest-level guard.
- #173 — caller-facing help surface (`lobes help <role>`, `GET /help`).
- #174 — `--force` must stop destroying operator-typed `.env` keys.
- #175 — warn when an operator profile shadows a built-in.
- Consumer coordination — the served-name change will 404 callers pinning the raw
  incumbent id (user decision: accept the break); a role-name migration issue is owed.
