# Delivery Summary — thor cortex speculation

plan: `thor-cortex-speculation` · run: `partial` · date: `2026-08-25`
baseline: `devague summary skeleton`

## Intent

Give the Jetson AGX Thor's `cortex` lane speculative decoding back. Since deviation d1
(2026-08-20) that lane has run with **no** speculation at all, because
`docs/evidence/2026-08-20-accept-cortex-local-thor.txt` recorded a flat "MTP MUST BE OFF on
sm_110" — the nightly ships no sm_110 image for `fused_gdn_decode_post_conv_mtp` and the lane
died on its first decode. This run set out to measure whether that is a wall or a config knob,
across three arms at 262144 (control / MTP-n2 / DSpark block-7), and to adopt whichever arm the
numbers chose — including "neither". The measurement was the deliverable either way.

## Planned Work

Quoted verbatim from the `devague summary` skeleton:

- `t1` — Build the rollback safety net and prove it works BEFORE touching anything
- `t2` — Re-render the deployment scaffold, preserving every operator-typed key
- `t3` — ARM A — the control: Triton GDN decode path, NO speculation, 262144
- `t4` — ARM B — MTP-n2: the unlock test
- `t5` — ARM C — DSpark block-7 against the pinned revision
- `t6` — Cross-arm output comparison and the divergence protocol
- `t7` — Write the evidence transcript — including if the answer is no
- `t8` — Adopt the outcome in-tree and prove the served contract
- `t9` — Ship it: branch, version bump, PR
- `t10` — Pre-fetch and verify the DSpark drafter BEFORE the maintenance window

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | Snapshot `~/.lobes.pre-speculation-20260825T054944Z`, then **exercised**: dir moved aside, restored, verified byte-identical, `drop_caches`, container recreated, re-probed. 12.1 → 12.2 tok/s, `known_answer` PASS. Box confirmed un-drifted. |
| `t2` | partial | Rendered with the repo CLI at `--apply --force` (d1, d2). Key-by-key `.env` diff: **zero keys lost**, every operator-typed peer key verbatim. Slot present (4 occurrences). No local `vllm-hand`. The `model=hand` proxy criterion **failed** for a pre-existing mesh reason (d3). |
| `t3` | delivered | ARM A booted with `GDN decode kernel: triton`, no spec flag in argv, KV 1,298,667 tok (4.95×). 12.19 / 12.19 / 12.21 tok/s; repeat drift 0.2%; warmup 11.36 discarded. |
| `t4` | delivered | ARM B booted MTP-n2 on the Triton path and **completed a multi-token decode**; zero `no kernel image` errors. 26.79 / 26.73 / 19.46 tok/s; acceptance 45.8–93.8%. |
| `t5` | blocked | DSpark **loads** (`DSparkDraftModel`, pinned revision in argv) and its KV **fits** (630,029 tok, 2.40×), then warmup dies with a CUDA illegal memory access in the draft attention path. Untestable hypothesis; no attention-backend knob exists (d4, #206). |
| `t6` | delivered | Probe suite on both runnable arms, 3 runs each. `known_answer` + `tool_calls` PASS on all 6. **A divergence was found** on 1 of 3 deterministic probes and investigated to the bounded stop (d5, #207). |
| `t7` | delivered | `docs/evidence/2026-08-25-spike-thor-cortex-speculation.txt` — 335 lines, 10 sections, every number carrying its conditions, both negative results written up with the same care as the positive one. |
| `t8` | delivered | `thor.toml`'s flat "MTP MUST BE OFF" replaced with the measured condition; `docs/qwen3.8-27b-nvfp4.md` + `docs/dspark-speculation.md` cite the transcript. h10 was **falsified then satisfied** — served contract now reports 262144 (d6, #208). |
| `t9` | delivered | Version 0.61.2 → 0.62.0, full gate green, PR **#210** opened. |
| `t10` | delivered | Drafter cached at the pinned revision, confirmed on disk via the `snapshots/85ef153b…` directory and a 2,718,576,122-byte blob. No partials; 285 GB free after. Ran in parallel with `t1`. |

## Mid-work Decisions

- `d1` — **t2 must use `--apply --force`, not the plain `--apply` its instruction preferred.** `docker-compose.yml` is `scaffold_action` "skip" without `--force`, and that file is the only thing that can introduce the slot t2 exists to obtain — while `_apply_profile_env` force-writes all 22 shape keys regardless. The plan's own instruction would have produced the worst available outcome: `.env` declaring DSpark @ 262144 while a slot-less compose kept serving 1M unspeculated.
- `d2` — **t2 must render with the repo CLI, not the box's installed 0.57.1**, which predates both the substitution slot (0.59.0) and the `speculative_config` knob (0.61.0). The operator upgraded to 0.61.2 mid-run, which resolved this but *not* d1.
- `d3` — **the `hand` referral is dangling**, for a mesh reason this render did not cause: the Spark now reports `hand: feasible=false` and 404s `role_infeasible`, contradicting the 2026-08-20 dual-cortex transcript's "hand STAYS served on the Spark".
- `d4` — **ARM C blocked**, and blocked on a *missing knob* rather than on DSpark.
- `d5` — **an output divergence was observed** between ARM A and ARM B, recorded before diagnosis exactly as h13 requires.
- `d6` — **h10 was falsified before it was satisfied**: the gateway advertised a stale served contract for the whole spike.
- `d7` — **recreating the gateway silently armed inbound auth**, because compose reads `GATEWAY_API_KEY` from the shell, not just `.env`. Restored to its prior posture rather than decided unilaterally.
- Not covered by any deviation record: **all three arms ran with `enable_thinking=false`**, uniformly, so they stay comparable and a small budget cannot die inside an unterminated `<think>`. The cortex role's real traffic uses thinking, so these are floor-and-multiplier figures, not a traffic simulation. Stated as a condition in the transcript.
- Not covered by any record: **an unrelated `uv.lock` change** (stripping `sys_platform` markers from the CUDA deps) was produced by running `uv` on this aarch64 box and was **backed out** of the commit rather than shipped.

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|-----------------------|----------------|
| `t2` (`d1`) | plain `--apply` cannot deliver the slot; `--force` required, and `.env` proved merge-only-always so the feared key loss did not occur | risky |
| `t2` (`d2`) | installed CLI 0.57.1 predates the slot and the knob; rendered with 0.61.2 instead | acceptable |
| `t2` (`d3`) | the `model=hand` proxy criterion cannot pass — the Spark no longer hosts `hand`; pre-existing mesh drift, unrelated to this render | needs-follow-up |
| `t5` (`d4`) | CUDA illegal memory access in the draft attention path; the FlashInfer hypothesis is untestable because the generate lanes expose no attention-backend knob | needs-follow-up |
| `t6` (`d5`) | greedy output diverges between the speculative and non-speculative arms; each arm internally stable; no cause proven, so the equivalence claim is retracted per h13's bounded clause | needs-follow-up |
| `t8` (`d6`) | capability surfaces advertised 1048576 against a 262144 lane until the gateway was recreated | needs-follow-up |
| `t8` (`d7`) | recreating the gateway armed inbound auth from the shell environment; restored, left for the operator to configure | risky |
| `c27` | the challenge pass predicted `--force` would flip `HAND_FEASIBLE` true. **Refuted by probe** — the thor card declares no `hand` role. The claim was corrected in-frame before execution rather than shipped wrong. | acceptable |

## Evidence

- transcript: `docs/evidence/2026-08-25-spike-thor-cortex-speculation.txt` (335 lines, 10 sections)
- offline suite: `uv run pytest -n auto` — **3182 passed, 15 skipped**
- profile goldens: `tests/test_profile_schema.py`, `tests/test_profile_render.py` — 79 passed
- live: `tests/test_live_capabilities.py` (`LOBES_SMOKE_BASE_URL=http://localhost:8000`) — 4 passed, 1 failed
- live: `tests/test_smoke_duo.py::test_live_main_text_returns_nonempty_content` — pass
- lint: `black` / `isort` / `flake8` clean; `bandit -c pyproject.toml -r lobes` — 0 low / 0 medium / 0 high
- rubric: `uv run afi cli doctor . --strict` — PASS
- commits: `b9d4fe5..1bc0cff` on `spec/thor-cortex-speculation`
- PRs / issues: **#210** (this run) · **#206**, **#207**, **#208**, **#209** (filed from findings)

## Delivery Claims

| Claim | Confidence | Evidence |
|-------|------------|----------|
| `VLLM_GDN_DECODE_KERNEL=triton` unlocks speculative decoding on sm_110 | high | ARM B booted and completed a multi-token decode with 0 `no kernel image` errors — transcript §3 |
| MTP-n2 delivers 26.79 tok/s on code vs a 12.19 tok/s control (+120%) | high | transcript §2–§3; repeat drift 0.0–0.2% within each arm |
| The Triton path costs essentially nothing on the unspeculated floor | high | 12.19–12.21 (Triton) vs 12.1–12.2 (CUDA) — transcript §1–§2. Resolves blocking unknown v1 |
| Rollback from this change is real and exercised, not assumed | high | transcript §1 — restore + recreate + re-probe reproduced 12.2 tok/s and `known_answer` PASS |
| `--force` re-render preserves every operator-typed peer key | high | key-by-key `.env` diff: zero keys removed, all `*_PEER_*` and `*_FEASIBLE` unchanged |
| The served contract now reports 262144 on both surfaces | high | `GET /capabilities` → `cortex context: 262144`, after the gateway recreate |
| Greedy output is NOT preserved under MTP on this lane | high | 1 of 3 deterministic probes diverges; each arm stable across 3 runs; 1.25-logprob gap at the divergence point — transcript §5, issue #207 |
| The divergence is caused by the GDN recurrent-state path differing under speculation | **unverified** | hypothesis only — the decisive test (ARM A logprobs at the same position) was not run. Explicitly not claimed |
| DSpark does not work on sm_110 | **unverified** | **NOT claimed.** ARM C is blocked on a missing knob (#206); viability is unknown in either direction |
| MTP-n2 would also fit at the 1M YaRN window | **unverified** | not tried; MTP's ~238k-token KV cost makes it plausible. Not claimed |
| These figures predict cortex's real throughput | **unverified** | all arms ran `enable_thinking=false`; real traffic uses thinking. Floor-and-multiplier figures only |

## Remaining Work / Follow-up

- **`t5` / #206** — add `--attention-config` to the generate lanes (mirroring the embed lanes) and thread it through the profile schema so the `SM_110` trait can reach them. Then re-run ARM C under `TRITON_ATTN`. The deeper bug is that a validated per-card attention divergence cannot currently be expressed for *any* generate lane.
- **`t6` / #207** — run the decisive logprob test on ARM A at the divergence position to separate "sampling/verification bug" from "the forward passes genuinely differ". One lane reboot.
- **`t8` / #208** — the gateway should derive the served contract from the lane rather than its own start-time env, or at minimum detect drift instead of advertising a window it cannot serve.
- **`t8` / #209** — decide the box's auth posture deliberately. The `.bashrc` export suggests an authenticated box is intended, but the deployment ran open for four days and `.env` says nothing. **Operator decision, deliberately left open.**
- **`t2` / d3** — repoint or clear the Thor's `HAND_PEER_ORIGIN`; the Spark no longer serves `hand`. Also: the Thor's gateway *hangs* rather than relaying the peer's 404, which may be its own defect.
- **Deployed gateway is 0.61.2 vs CLI 0.62.0** — the one failing live test is honest drift detection. `lobes fleet up --apply` rebuilds it. Not silenced.
- **Parked, not blocking**: issue #204's shape-precedence gap (a Thor-specific speculative config has no ergonomic home); retiring the YaRN `hf_overrides` at 262144 (its own unmeasured axis); no first-class draft-acceptance metric in `lobes/_metrics.py`, so an adopted speculative lane degrades invisibly.
