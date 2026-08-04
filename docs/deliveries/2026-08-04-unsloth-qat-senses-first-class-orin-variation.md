# Delivery Summary — unsloth QAT senses + first-class orin variation

plan: `unsloth-qat-senses-first-class-orin-variation` · run: `partial` · date: `2026-08-04`
baseline: `devague summary skeleton`

## Intent

> senses serves unsloth/gemma-4-12B-it-qat-w4a16, live-validated on this Jetson AGX Orin as a first-class orin variation, while the thor/spark machine profiles stay intact so moving the setup to another architecture stays a profile pick, not a rework

This run executed that plan's 13 tasks via `/assign-to-workforce`: three wave-0
repo tasks fanned out to isolated worktree agents in parallel, two box-side
measurement tasks run by the main agent, then three sequential worktree agents
for the profile/shape/compose layers. **All eight repo-side and measurement
tasks delivered; the five tasks that require mutating the live deployment are
blocked** — see Drift and Remaining Work.

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
| `t9` | blocked | Re-render fully rehearsed against copies and proven correct, but `lobes init --apply --force` was denied by the environment's permission classifier. No boot, no measured budget. (`d7`) |
| `t10` | blocked | Depends on `t9`. Peer side verified reachable and keyless; this box's gateway rebuild is the prerequisite. (`d5`, `d7`) |
| `t11` | blocked | Probe harness **built and validated against the incumbent**, but never run against the new checkpoint. (`d7`) |
| `t12` | partial | Evidence transcript committed with the incumbent baseline, capability baseline, streaming contract, pre-boot checkpoint facts and projection. The **measured-value backfill** of `orin.toml`/`orin-lobe.toml` did not happen. (`d7`) |
| `t13` | partial | Version bumped to 0.55.0, full suite green, untouched-proof verified, five follow-up issues filed. PR opened carrying the repo-side work only. |

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
- `d7` — the live re-render was **denied by the environment's permission classifier**,
  blocking every remaining box-side task.
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
| `t9`, `t11`, `t12` (`d7`) | live mutation denied by the permission classifier — no boot, no measured budget, no probe run against the new checkpoint | needs-follow-up |

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
| **senses serves the unsloth checkpoint on this box** | unverified | `t9` blocked — NOT claimed done |
| **the 0.45 / 262144 budget holds on real hardware** | unverified | MEASURED-PENDING hypothesis; no boot occurred |
| **image / video / audio / reasoning on the new checkpoint** | unverified | probe harness built and validated against the *incumbent* only |
| **worker is reachable from this box via the Thor** | unverified | Thor side verified reachable + keyless; this box's gateway still 0.45.0 |

## Remaining Work / Follow-up

- `t9` — an operator must run the re-render (the exact command and its safeguards are
  in the PR description). Boot **senses first** — boot order materially changes the KV
  pool. Record both the value that booted **and** any value refused.
- `t9` follow-on — restore the 12 captured operator `.env` lines immediately after
  `--force`, then verify `model=cortex` still answers with `X-Lobes-Proxied-By`.
- `t11` — run the probe harness against the new checkpoint and compare to the
  incumbent baseline already recorded. Remember `enable_thinking` or reasoning will
  falsely read as absent.
- `t12` — backfill the measured budget into `orin.toml` **and** `orin-lobe.toml`; a
  lockstep test fails CI if only one moves.
- `t10` — after the gateway rebuild, add `WORKER_PEER_ORIGIN` + `WORKER_PEER_PROXY=true`
  (no credential needed) and verify `model=worker` answers with `X-Lobes-Proxied-By`.
- #171 — correct `docs/orin-profiles.md`'s KV figure and strengthen the boot-ordering caveat.
- #172 — the host-dependent-test class is fixed for two files; consider a conftest-level guard.
- #173 — caller-facing help surface (`lobes help <role>`, `GET /help`).
- #174 — `--force` must stop destroying operator-typed `.env` keys.
- #175 — warn when an operator profile shadows a built-in.
- Consumer coordination — the served-name change will 404 callers pinning the raw
  incumbent id (user decision: accept the break); a role-name migration issue is owed.
