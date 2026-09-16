# Delivery Summary — innereye: lobes hosts ComfyUI

plan: `innereye-lobes-hosts-comfyui` · run: `complete` · date: `2026-09-16`
baseline: `devague summary skeleton`

## Intent

Issue #268, filed by the sibling `innereye` project, asked lobes to take over
ComfyUI's lifecycle, give it a slot in the unified-memory budget, and front it
on the network behind the gateway's auth — and explicitly asked lobes to
*decide* whether ComfyUI is a first-class Colleague role or a different class
of tenant, "rather than let it emerge". This run executed the converged plan
that answers that: `innereye` is the eleventh role, served by a containerized
ComfyUI behind a job-scoped `/v1/render` facade.

## Planned Work

Quoted verbatim from the `devague summary` skeleton:

- `t1` — Declare the innereye/cortex `exclusive_roles` group and prove lobes init refuses the bad shape
- `t2` — Dockerfile.comfyui: reproduce the working install as a locally-built image
- `t3` — Compose service skeleton: expose-only isolation, healthcheck, logging, GPU access
- `t4` — Compose volumes and ownership: read-only weights, writable output, non-root user
- `t5` — Register innereye as the eleventh role ATOMICALLY: every role-keyed table, declaration-only
- `t6` — Profile and shape wiring: `INNEREYE_`\* env family and the declared-peak co-residency veto
- `t8` — Gateway plumbing: a method-general upstream opener and a GET-side streaming relay
- `t9` — The /v1/render facade: job-scoped routes, fan-out, and the auth placement
- `t10` — Lifecycle: the up target, the cold-backend 503, and the no-actuator guard
- `t11` — Capabilities and mesh advert for the render lane
- `t12` — The reference doc, the non-goals, and the catalog entry
- `t13` — Live acceptance run on the DGX Spark and its evidence transcript

Two further tasks, `t7` (goldens) and `t14` (role vocabulary), were **rejected
during planning** — they were absorbed into `t5` under approved deviation `d2`.
Both appear under Drift below rather than silently vanishing.

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | `[[exclusive_roles]]` on `spark.toml` + the new `spark-innereye` shape + 4 goldens; merged `29515ae` |
| `t2` | delivered | `Dockerfile.comfyui`, BUILT and render-validated live (41.84s vs the baseline's 42.05s); merged `577a519` |
| `t3` | delivered | the `comfyui` compose service, expose-only, `/object_info` healthcheck; merged `5cd5cc2` |
| `t4` | delivered | read-only models mount, writable output, `user: 1000:1000`; verified through the service; merged `d40da75` |
| `t5` | delivered | **re-scoped by `d2`** to an atomic registration absorbing `t7` + `t14`; merged `46feb80` |
| `t6` | delivered | `declared_peak_gib` lane-gated to innereye, single consumer; merged `6fcd2f6` |
| `t7` | dropped | absorbed into `t5` by `d2` — registering a role cannot be split file-disjoint |
| `t8` | delivered | method-general `open_upstream` + GET-side streaming relay; merged `23b7416` |
| `t9` | delivered | six allowlisted `/v1/render` spellings, job-scoped; merged `8d79a35` |
| `t10` | delivered | `lobes up innereye`, honest 503, AST import guard; merged `4d0e982` |
| `t11` | delivered | per-backend readiness path + mesh non-wiring; merged `0ce2967` |
| `t12` | delivered | `docs/comfyui-innereye.md`; colleague-stack ten → eleven roles; merged `2ae7a30` |
| `t13` | delivered | `docs/evidence/2026-09-16-accept-innereye-spark.txt`, incl. cross-box from the Thor; `8d04f09` |
| `t14` | dropped | absorbed into `t5` by `d2` |

## Mid-work Decisions

- `d1` — the plan assumed ONE role registry; there are TWO. `lobes.roles.ROLES`
  (Colleague-facing) and `lobes.profiles.schema.ROLES` (profile machinery) are
  separate constants and `schema.py:317` validates `[[exclusive_roles]]`
  against the second. The brief given to `t1` named the wrong one. Carved a
  role-vocabulary task and reordered.
- `d2` — registering a role is **atomically cross-cutting** and cannot be
  decomposed file-disjoint: `t5` merged correctly within its file boundary and
  left 9 failures, all of them the repo's OWN invariant tests asserting a role
  is registered everywhere or nowhere. `t5` was re-scoped to absorb `t7` and
  `t14`. Cost accepted: one wide task instead of four narrow ones, because the
  alternative was a knowingly-red main branch.
- `d3` — the gateway's readiness probe had no owner for the render lane.
  `_readiness.py` hardcoded `/health`, which ComfyUI 404s (MEASURED live), so a
  wired innereye would have advertised `feasible:true, ready:false` forever,
  contradicting the plan's own success signal.
- **Operator decision, not covered by a deviation record:** v1 declares
  `cortex`/`innereye` mutually exclusive rather than measuring co-residency
  first. This turned the blocking risk `r1` from a gate into a scoping
  decision, and was later *confirmed* by measurement rather than merely assumed.
- **Operator decision:** downtime accepted (innereye is experimental, not
  productized), so the two repos were not sequenced to avoid a gap.
- **Operator decision:** the facade is job-scoped rather than a prefix relay,
  accepting that innereye needs a client change — not the config change #268
  assumed.

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|-----------------------|----------------|
| `t1` (`d1`) | the brief named the wrong role registry; `t5` landing did not unblock it, and a second blocker (`ROLE_SERVICE`) sat behind it | needs-follow-up |
| `t5` (`d2`) | registering a role is atomically cross-cutting; the planned file-disjoint decomposition fought an invariant the codebase already enforces | needs-follow-up |
| `t7`, `t14` (`d2`) | absorbed into `t5`; dropped as separate tasks | acceptable |
| `t11` (`d3`) | gained the gateway-side readiness probe, which no task owned | acceptable |
| `t9` | also added the `INNEREYE_BASE_URL` reader and gateway compose passthroughs, outside its stated boundary, because the reader is inert without them | acceptable |
| `t13` | the gateway under test was merged **source**, not the packaged image — `Dockerfile.gateway` installs `lobes-cli` at `MODEL_GEAR_VERSION` and 0.79.0 is unpublished | needs-follow-up |

Both `d1` and `d2` trace to one root cause, filed as issue **#269**: lobes
carries two silently divergent role registries. `t14` exists only because of
it and should be deletable once #269 lands.

## Evidence

- tests: full suite `uv run pytest tests/ -q -n auto` — **5151 passed, 15 skipped**
- tests: obligation surfaces (8 files, incl. `tests/test_gateway_render_facade.py`, `tests/test_gateway_no_lifecycle_actuator.py`, `tests/test_comfyui_volumes.py`) — **304 passed**
- lint: `black --check`, `isort --check-only`, `flake8`, `bandit` — clean
- commits: `9f7efdd..8d04f09`
- live acceptance: `docs/evidence/2026-09-16-accept-innereye-spark.txt`
- live baseline: `docs/evidence/2026-09-16-baseline-comfyui-venv-spark.txt`
- live container comparison: `docs/evidence/2026-09-16-t2-comfyui-container-spark.txt`
- evidence records: `e1`–`e12` (all `proposed`, LLM-origin — pending human confirm)
- behavioral deltas: `b1`–`b7` · lapses: `l1`–`l3`
- issues: lobes-cli **#268**, **#269**; innereye **#5**

## Delivery Claims

| Claim | Confidence | Evidence |
|-------|------------|----------|
| `innereye` is registered as the eleventh role across every role-keyed table | high | `e5` · full suite 5151 passed · commit `46feb80` |
| The service publishes no host port; 8188 is refused from the host and from the Thor | high | `e3` · acceptance §2 (both controls re-run after a confound was removed) |
| An unauthenticated request to `/v1/render` is refused 401, with an attribution control | high | `e8` · acceptance §3 |
| `GET /capabilities` reports innereye feasible **and** ready | high | `e10` · acceptance §4 · `tests/test_gateway_readiness.py` |
| A render completes end-to-end through the facade, artifact bytes byte-identical to disk | high | `e8` · acceptance §5 · sha256 equality |
| The Thor renders and fetches across the tailnet, byte-identical | high | `e12` · acceptance §7 · sha256 `49515180…` |
| Artifact enumeration across jobs is refused | high | `e8` · acceptance §6 · `tests/test_gateway_render_facade.py` |
| The declared peak meters nothing — its only consumer is the co-residency veto | high | `e6` · source-scanning test in `tests/test_init_coresidency.py` |
| The data plane cannot start or stop anything | high | `e9` · `tests/test_gateway_no_lifecycle_actuator.py` (AST guard + control) |
| Weights stay reachable outside Docker; artifacts are host-owned | high | `e4` · verified through the compose service, not `docker run` |
| comfy-aimdo is cgroup-blind — the declared peak is not a co-residency budget | high | `e2` · container figures identical to bare metal |
| ComfyUI serializes overlapping submissions; peak does not double | **low** | `l3` (approved-pending) — n=1 pair, one graph, video path never exercised |
| The packaged gateway image builds and runs at 0.79.0 | **unverified** | not tested — 0.79.0 unpublished; merged source was used instead |
| innereye-the-client can drive the facade | **unverified** | not tested — every request was curl; innereye#5 tracks adoption |
| The Wan 2.1 14B video path works through the lane | **unverified** | never exercised — and it is the larger of the two paths |
| A memory-limited cgroup changes nothing | **unverified** | plan risk `r3` — every measurement ran with no `--memory` cap |

## Remaining Work / Follow-up

- **#269 — one set of roles.** The root cause of `d1` and `d2`. Until it lands,
  adding a role stays a scavenger hunt across undocumented tables.
- **innereye#5 — facade adoption.** The client speaks raw ComfyUI and has no
  bearer support. No urgency: downtime accepted, venv rollback intact.
- **Packaged gateway image at 0.79.0** — verify once published; the acceptance
  transcript names this as untested.
- **`r3` — memory-limited cgroup** — the first thing to try if co-residency is
  ever revisited.
- **Frame vagueness `v3`** — whether the render lane participates in the
  pressure shed matrix. Deliberately unwired: shedding a resident renderer
  refuses work without freeing memory.
- **`deployments/` catalog entry** — deferred deliberately; no box's lock has
  been captured for this shape, and fabricating one was declined.
- **Wan 2.1 video path** — unexercised, larger footprint than FLUX.
- **Evidence/delta/lapse records `e1`–`e12`, `b1`–`b7`, `l1`–`l3`** are all
  `proposed` and need the gate-owning human's confirm.
