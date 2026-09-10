# Delivery Summary — thor worker arm: Qwen3.6-35B-A3B recipes

plan: `thor-worker-arm-qwen3-6-35b-a3b-recipes` · run: `partial` · date: `2026-09-10`
baseline: `devague summary skeleton`

## Intent

Issue #244 asked for the Jetson AGX Thor's `worker` lane to be rebuilt around a
Qwen3.6-35B-A3B NVFP4 checkpoint on a **Thor-measured** recipe rather than a
copied one, with the box giving up its local Qwen3.8 `cortex` to pay for it.
The operator raised the bar mid-frame from "recover the old 61.2 tok/s" to a
**100 tok/s target**, and the run executed the converged plan's fifteen tasks
across six waves.

## Planned Work

Quoted verbatim from the `devague summary` skeleton:

- `t1` — Catalog: add nvidia/Qwen3.6-35B-A3B-NVFP4, demote Nemotron off the worker seat
- `t2` — Give associate its own catalog resolution, independent of worker
- `t3` — Wire the recipe knobs the worker lane cannot express today
- `t4` — Re-widen the worker role contract to multimodal coder
- `t5` — Extend the spec-arm harness: per-position acceptance, single-stream vs aggregate
- `t6` — Clear the gateway's pool-arming trap and the stale worker default
- `t7` — Decide and record the pressure floor for a cortex-less Thor
- `t8` — Rollout note and raw-id consumer audit, published BEFORE the flip
- `t9` — Capture the pre-flip baseline on the Thor — the numbers that become unrecoverable
- `t10` — Back up the deployment dir and prove the rollback path
- `t11` — Live spike on the Thor: does the checkpoint load, and which MoE backend boots
- `t12` — MTP depth sweep on the Thor: off / 1 / 3 / 5 / 7, acceptance per position
- `t13` — The flip: re-render the Thor, retire the cortex pool, probe the mesh
- `t14` — The fast gear: earn 100 tok/s or report the measured best
- `t15` — Re-point the thor-worker shape at the measured checkpoint and budget

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | `nvidia/Qwen3.6-35B-A3B-NVFP4` added with metadata re-read from the checkpoint's own `config.json`/`hf_quant_config.json`; Nemotron demoted. Merged `483f2f2`. |
| `t2` | delivered | associate given its own catalog resolution; needed three fixes, not one — the catalog hint, `roles.py`, and the gateway's `_PEER_ROLE_HINT` proxy table + `_DEFAULT_ASSOCIATE`. Merged `1b0aa1f`-series. |
| `t3` | delivered | ten knobs wired end to end (schema → render → compose → env.example → goldens), byte-identity proved against a real `docker compose config`. Merged `40f740b`. |
| `t4` | delivered | `image_understanding`/`video_understanding` restored, `code_authoring` removed from forbidden; video labelled declared-not-measured. Merged. |
| `t5` | delivered | spec-arm harness captures vLLM's per-position acceptance array and separates single-stream from aggregate legs. Merged `eb89e15`. |
| `t6` | delivered | `lobes doctor` gained a `pool_arming` finding reusing the gateway's own guard offline; `_DEFAULT_WORKER` derived from the catalog. Merged `94061c7`. |
| `t7` | delivered | decided and documented: the floor's absence is declared, not papered over — `docs/evidence/2026-09-10-thor-pressure-floor-244.txt`. |
| `t8` | delivered | rollout note + raw-id audit published before the flip; corrected post-merge when `t1` landed the real checkpoint id (`dcc360d`). |
| `t9` | delivered | pre-flip baseline captured — 18.7 tok/s, and it caught that the committed docs were stale on three counts. `29aa5e1`. |
| `t10` | delivered | `~/.lobes` backed up byte-identical with SHA-256s; reverse-render dry-run proved the render does **not** restore. `446254c`. |
| `t11` | delivered | the checkpoint loads on sm_110; Marlin MoE + FlashInfer auto-selected; budget measured. `eb27650`. |
| `t12` | delivered | five-arm sweep (off/1/3/5/7) plus a k=6 arm added at the operator's request. `04cce66`. |
| `t13` | **partial** | the flip is live and serving; `model=cortex` proxies to the Spark. **The cross-box leg is not delivered** — no peer declares a `WORKER_PEER_ORIGIN` pointing at the Thor. `c337521`. |
| `t14` | delivered | DFlash measured and deployed at 196.6 tok/s through the gateway; no custom vLLM build needed. `c337521`. |
| `t15` | delivered | shape re-pointed at the measured checkpoint/budget; two goldens moved, four keys each. Merged. |

## Mid-work Decisions

- `d1` — `t8` was briefed with the nvidia checkpoint but could not find it in the catalog (`t1` had not merged yet), so it resolved the target to the unsloth sibling and flagged the inference. Corrected post-merge; the unsloth recipe kept as a labelled fallback. *A wave-0 ordering gap: the dependency was real but not expressed in the graph.*
- `d2` — the plan's stated rollback ("re-render the previous shape and restore the saved `.env`") is not sufficient alone: `lobes init` is merge-only for `.env` and leaves compose untouched without `--force`. The backup **is** the rollback.
- `d3` — the pressure-floor question was answered by probing rather than re-measuring `hand`: `model=hand` is unservable from **every** box in the mesh today (misdeclared referral), so the floor's absence is pre-existing, not created by this flip.
- `d4` — `t13`'s cross-box acceptance criterion is **not met**: no peer declares a `WORKER_PEER_ORIGIN` pointing at the Thor. Wiring the Spark/Orin needs their own `.env` and this box has no SSH access (publickey denied).
- `d5` — the deployed lane serves the **full native 262144** window, not the 65536 the spike measured and `t15` committed: a real consumer (Qwen Code, requesting 64000 output tokens) returned HTTP 400 against the smaller window.
- `d6` — the deployed `docker-compose.yml` was hand-patched with two of `t3`'s knob slots rather than re-scaffolded, because the file is hand-edited (the #214 drift condition `c29` names).
- Operator decision mid-run: **k=6 was added to the sweep** on the operator's observation that k=7 showed an acceptance drop. It proved to be the MTP peak (156.3 tok/s at 87.2% acceptance), dominating k=7 on both axes.
- Operator decision mid-run: **DFlash was selected as the deployed profile** over MTP, on the measured 196.6 vs 156.3 tok/s gap.

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|-----------------------|----------------|
| `t8` (`d1`) | briefed with a checkpoint id that did not yet exist in the catalog; resolved to the wrong sibling and flagged it | acceptable |
| `t10` (`d2`) | the render is merge-only, so the documented rollback path was incomplete as written | acceptable |
| `t7` (`d3`) | answered by probing rather than re-measuring a known-broken lane on an un-upgraded engine | needs-follow-up |
| `t13` (`d4`) | no SSH access to the peer boxes; the cross-box acceptance criterion cannot be met from here | needs-follow-up |
| `t13`/`t15` (`d5`) | a consumer broke on the 65536 window, so the deployed window is 262144 and the shape TOML is now behind deployed reality | needs-follow-up |
| `t13` (`d6`) | the deployed compose could not express the measured recipe until `t3`'s knob was patched in by hand | acceptable |
| `t12` | the plan specified k = off/1/3/5/7; a k=6 arm was added mid-run at the operator's request and became the MTP peak | acceptable |
| `t14` | the plan treated DFlash as a downstream spike gated on `t13`; it was measured before the flip and became the deployed default | acceptable |

## Evidence

- tests: full suite `uv run pytest -q -n auto` — **4475 passed, 15 skipped** at `c337521`
- tests: `tests/test_catalog.py tests/test_associate_role.py tests/test_worker_recipe_knobs.py tests/test_shape_goldens.py` — 203 passed
- lint: `uv run black --check lobes tests` — 280 files unchanged; `flake8` clean
- commits: `c8565b8..c337521` (13 commits: 7 merges, 6 evidence/doc)
- version: `0.73.7 → 0.74.3`
- live transcripts: `docs/evidence/2026-09-10-baseline-thor-cortex-pre-244.txt`,
  `…-thor-backup-and-rollback-dryrun-244.txt`, `…-thor-pressure-floor-244.txt`,
  `…-spike-nvidia-35b-a3b-preboot-thor.txt`, `…-accept-nvidia-35b-a3b-thor.txt`,
  `…-sweep-mtp-depth-nvidia-35b-a3b-thor.txt` (+ raw log), `…-accept-thor-worker-flip.txt`
- obligations/evidence filed via `/validate-delivery`: `o1`–`o9`, `e1`–`e10` (all **proposed**, pending adjudication)
- issue: #244

## Delivery Claims

| Claim | Confidence | Evidence |
|-------|------------|----------|
| the Thor serves `nvidia/Qwen3.6-35B-A3B-NVFP4` as `worker` at the native 262144 window | high | `docs/evidence/2026-09-10-accept-thor-worker-flip.txt` §1–2 · evidence `e1` |
| single-stream decode is **196.6 tok/s** through the gateway, vs a 100 target and 61.2 floor | high | same transcript §3 · `scripts/stream-measure.py` · evidence `e2` |
| the checkpoint loads on sm_110; Marlin MoE + FlashInfer are auto-selected | high | `…-accept-nvidia-35b-a3b-thor.txt` §1–2 · evidence `e6` |
| MTP acceptance and per-position rates are measured across k = off/1/3/5/6/7 | high | `…-sweep-mtp-depth-nvidia-35b-a3b-thor.txt` + raw log |
| tool calls parse into `tool_calls` and vision passes with a negative control | high | flip transcript §4 · evidence `e8` |
| Qwen Code drives the lane end to end (read → `write_file` → correct output) | high | flip transcript §5 |
| every added knob is reachable end-to-end with byte-identical defaults | high | `tests/test_worker_recipe_knobs.py` · evidence `e5` |
| associate's default survives a worker checkpoint promotion | high | `tests/test_associate_role.py::test_associates_default_survives_a_worker_checkpoint_promotion` |
| `model=cortex` still answers from the Thor, forwarded to the Spark | high | flip transcript §4 (200 + `X-Lobes-Proxied-By`) |
| **a peer box reaches the lane as `model=worker`** | **unverified — FAILS** | evidence `e3` records the Spark advertising the stale Nemotron id, `feasible=false`. Not claimed done. |
| DFlash beats MTP under **concurrency** | unverified | only batch-1 measured; no concurrent leg run at any setting |
| the 262144 window retrieves correctly at depth | unverified | the window boots and serves; no long-context retrieval probe run |
| video intake works | unverified | declared by the checkpoint; only image intake was probed |

## Remaining Work / Follow-up

- **`t13` cross-box leg (`d4`)** — declare `WORKER_PEER_ORIGIN`/`_PEER_PROXY`/`_PEER_API_KEY` on the Spark (and the Orin) pointing at `http://thor.tail0be7e0.ts.net:8000`, then re-probe for `X-Lobes-Proxied-By`. **Operator action on those boxes** — this box has no SSH access.
- **`thor-worker.toml` is behind deployed reality (`d5`)** — it commits `max_model_len=65536`; the box serves 262144. Re-point the shape and regenerate its goldens.
- **The deployed compose is hand-patched (`d6`)** — reconcile `~/.lobes/docker-compose.yml` with the packaged template, or capture it as a `deployment.lock.toml` variation (#214).
- **`hand` is unservable mesh-wide (`d3`)** — the Thor's `HAND_PEER_ORIGIN` points at the Spark, which declares it infeasible; only the Orin declares it feasible and its lane is not running. Repoint and start it, or document the floor's absence mesh-wide.
- **Concurrency is entirely unmeasured** — every arm is batch 1. The high-k and DFlash wins rest on idle GPU capacity absorbing rejected draft tokens; under load that advantage may invert. This is the single largest untested assumption in the delivery.
- **`o1`–`o9` / `e1`–`e10` await adjudication** — filed `proposed` by design; `devague oblige --confirm` / `evidence --confirm` are the human's move.
- **Deviations `d4`–`d6` await confirmation**, which also unblocks three filed-but-refused behavioral deltas.
