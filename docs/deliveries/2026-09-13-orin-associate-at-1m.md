# Delivery Summary — orin associate at 1M

plan: `orin-associate-at-1m` · run: `complete` · date: `2026-09-13`
baseline: `devague summary skeleton`

## Intent

> The Jetson AGX Orin's associate lane serves Nemotron 3.5 Lightning at the native 1,048,576-token window as shipped, measured lobes configuration — not hand-typed .env keys — using the checkpoint that wins a measured NVFP4-vs-W4A16 A/B on the Orin

After: rendering orin-associate on the Orin produces a lane that boots at `max_model_len` 1048576 with the A/B-measured checkpoint, util, batched-token cap and `max_num_seqs`; the gateway advertises context 1048576; and every value cites a docs/evidence transcript

The run executed plan `docs/plans/2026-09-13-orin-associate-at-1m.md` (split: `docs/plans/2026-09-13-orin-associate-at-1m-split.md`) on branch `feat/orin-associate-1m`, tracked in issue #260. The A/B chose the checkpoint: NVIDIA NVFP4 was kept because W4A16 missed the ≥10% rule.

## Planned Work

Quoted verbatim from the `devague summary` skeleton:

- `t1` — associate lane knobs: `ASSOCIATE_MAX_NUM_SEQS` argv and `VLLM_ALLOW_LONG_MAX_MODEL_LEN` passthrough
- `t2` — pin one associate gear: `test_exactly_one_associate_gear`
- `t3` — measure transcript: docs/evidence/2026-09-13-measure-associate-budget-orin-1m.txt
- `t4` — Orin card: `GATEWAY_READ_TIMEOUT`=7200 in \[`host_env`\]
- `t5` — live rollout on the Orin and accept transcript docs/evidence/2026-09-13-accept-orin-associate-1m.txt
- `t6` — orin-associate shape at 1M: hosts, header, measured budget, rollback note; card doc block in lockstep
- `t7` — docs reconciliation to the measured 1M values

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | `vllm-associate` gains the conditional `--max-num-seqs` token and the `VLLM_ALLOW_LONG_MAX_MODEL_LEN` passthrough, plus `env.example` entries, tests and the template-defaults golden. Commits `c2b4170` and `1815dca` (d1), merged in `1db8387`. |
| `t2` | delivered | `test_exactly_one_associate_gear` and its duplicate-detection companion, test-only; `lobes/catalog.py` unchanged. Commit `c5a21a3`, merged in `52af2ef`. |
| `t3` | delivered | `docs/evidence/2026-09-13-measure-associate-budget-orin-1m.txt`: both arms, every metric, and a verdict that applies c24. Commit `f11435c`. |
| `t4` | delivered | `lobes/profiles/builtin/orin.toml` `[host_env]` `GATEWAY_READ_TIMEOUT = "7200"`, `tests/test_orin_gateway_read_timeout.py`, and 10 Orin goldens each +1 line. Commit `e9775cb`, merged in `9c7d990`. |
| `t5` | delivered | Live rollout on the physical Orin (06:00–07:29Z): all acceptance checks passed. `docs/evidence/2026-09-13-accept-orin-associate-1m.txt`, commit `45b3978`. |
| `t6` | delivered | `orin-associate.toml` hosts associate + embedder + reranker, with util 0.70 / 1048576 / 8192 / max_num_seqs 2, the associate-first boot note and a ROLLBACK note; card doc block in lockstep; tests and goldens. Commit `8bf878d` plus a main-agent review fix `728dbc8`, merged in `6eeacca`. |
| `t7` | delivered | The five docs reconciled to 1M, citing both transcripts, with the 128K figures kept as history. d3 also updated the shape header and summary and the pinning test. Commits `2d37a97` and `5fe896b`, merged in `ce7b120`. |

## Mid-work Decisions

- `d1` — widen t1 by one test file: recompute ONLY the vllm-associate entry in `tests/test_tool_parser_plugin.py` `_EXPECTED_NON_PRIMARY_HASHES` with a dated comment. Reason: t1's required vllm-associate change moves that service's hash, which failed the tool-parser plan's non-primary service hash lock; the file documents recomputing on a deliberate change, with precedent. Operator approved 2026-09-13.
- `d2` — t7 docs quote the shipped shape's minimum available host memory next to the A/B's 2,588 MiB, instead of 2,588 MiB alone. Reason: the live rollout measured lower headroom, and quoting only 2,588 would overstate it on a zero-swap board. Operator approved 2026-09-13.
  - **Number correction (same approved rule):** the d2 record quotes 1,952 MiB, the step-9 minimum. The minimum over the whole run was **1,925 MiB** (07:09:49Z). The docs use 1,925. The correction was posted on #260 (comment 5651996338).
- `d3` — widen t7: the `orin-associate.toml` header and summary go from DECLARED/UNVALIDATED to VALIDATED live 2026-09-13, for exactly what the accept transcript shows, and the pinning test now asserts that citation. Reason: the accept transcript booted the rendered shape on the physical Orin, which meets the shape's own "until one lands" condition. Operator approved 2026-09-13.
- **Main-agent review of t6** — before merging, fixed three claims and added one missing piece, all comment-only in `orin-associate.toml` (commit `728dbc8`):
  - a false "same quadruple as the live `.env`" before-state claim;
  - the memory headroom, which was attributed to associate alone;
  - a "zero container restarts" line that contradicted the embed restart;
  - the explicit ROLLBACK note, which was missing.
  No deviation record: the change stayed within t6's acceptance criteria.
- **Rollout approach (t5)** — the Orin gateway is recreated from its existing local image with `lobes up gateway --apply`, with no rebuild. Plan risk r2 was resolved with that decision.
  - `.env` was backed up and `~/.lobes` tarred before rendering, because `lobes init --force` overwrites scaffold files without a backup (plan risk r1, amended with the cited behaviour).
  - Boot order: associate first, then the gears, then the gateway.
- **Re-scaffold side effect (t5), no deviation record** — `--force` brought the Orin's stale compose current. That gave the reranker the #227 judge-prompt `--chat-template`, a live scoring change on the Orin. It was recorded in the accept transcript (step 4) and as delta `b4`.
- **Validation evidence** — e19 recorded a false **fail**, produced by a line-scoped grep that couldn't see an enclosing HISTORY heading or a wrapped Spark budget sum. e21 re-checked each flagged line in context (pass). Operator recommendation: reject e19.

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|------------------------|-----------------|
| `t1` (`d1`) | t1's required vllm-associate change (`ASSOCIATE_MAX_NUM_SEQS` argv + `VLLM_ALLOW_LONG_MAX_MODEL_LEN` env) moves that service's hash, failing the tool-parser plan's non-primary service hash-lock (1 failed / 4952 passed); the file documents recompute-on-deliberate-change and has precedent (#120, #222, #227, #217). Operator approved 2026-09-13. | `acceptable` |
| `t7` (`d2`) | the live rollout (t5 step 9, cold 1,040,073-token request through the Orin gateway) measured minimum available memory 1,952 MiB at 06:11:19Z on the rendered shape with embed, rerank and gateway resident, lower than the A/B's 2,588 MiB; quoting only 2,588 would overstate headroom on a zero-swap board. Operator approved 2026-09-13. | `acceptable` |
| `t7` (`d3`) | the accept transcript booted the rendered shape on the physical Orin, meeting the shape's own 'until one lands' condition; t7's docs correctly call it accepted/validated while the merged shape file and its pinning test still claim no box booted it. Neither t6 nor t7 covered those files. Operator approved 2026-09-13. | `acceptable` |
| `t5` | Re-scaffolding changed the Orin reranker's behaviour (the #227 chat template), which the plan did not anticipate. Recorded in the accept transcript and as delta `b4`; no deviation record covers it. | `risky` |

## Evidence

- tests: full suite at `9a251fa` (2026-09-13T07:55:50Z) — `4958 passed, 15 skipped` (pass)
- tests: the 187-test obligation run at `b6a287b` (07:52:44Z) covering `tests/test_associate_compose.py`, `tests/test_associate_exposure.py`, `tests/test_catalog.py::test_exactly_one_associate_gear`, `tests/test_orin_gateway_read_timeout.py`, `tests/test_orin_associate_shape.py`, `tests/test_tool_parser_plugin.py::TestOtherServicesUntouched`, `tests/test_shape_goldens.py`, `tests/test_profile_goldens.py`, `tests/test_gateway_fleet_doc.py` — `187 passed` (pass)
- TDD merge gate: full suite before and after each merge — t2 4947→4949, t1 4949→4955, t4 4955→4958, t6 4958→4958, t7 4958→4958 (all pass)
- lint at `9a251fa`: `uv run black --check lobes tests` clean; `isort --check-only` clean; `flake8` clean; `bandit -c pyproject.toml -r lobes` clean; `markdownlint-cli2` clean on the touched docs
- live: `docs/evidence/2026-09-13-accept-orin-associate-1m.txt` (rollout at `6eeacca`, 06:00:18Z–07:29:26Z) and `docs/evidence/2026-09-13-measure-associate-budget-orin-1m.txt` (A/B, 02:22–04:33Z)
- delivery ledger (`.devague/deliveries/orin-associate-at-1m.json`, all `llm`-origin and proposed): obligations `o1`–`o15`, evidence `e1`–`e21`, deltas `b1`–`b4`
- commits: `0145a54..9a251fa` on `feat/orin-associate-1m` (stacked on PR #258's `evidence/lightning-thor-v029`)
- PRs / issues: #260 (tracking; deviations and results in comments 5651326251, 5651445280, 5651806426, 5651996338, 5652044937), #258 (Thor evidence, unmerged, base of this branch)

## Delivery Claims

No lapses are filed for this plan (`devague lapse --list`: none). All ledger evidence is still **proposed**, so these confidence levels rest on the tests and transcripts cited, not on adjudicated ledger records.

| Claim | Confidence | Evidence |
|-------|------------|----------|
| the associate compose lane renders `--max-num-seqs` only when `ASSOCIATE_MAX_NUM_SEQS` is set, and passes `VLLM_ALLOW_LONG_MAX_MODEL_LEN` (default 0) | high | test `tests/test_associate_compose.py::TestDeclaredKnobsRenderNoDeadDeclarations::test_max_num_seqs_set_renders_flag` · `::TestAssociateAllowLongMaxModelLen` · evidence `e1`, `e3` |
| the associate exposure contract is unchanged | high | `tests/test_associate_exposure.py` passes unmodified (git diff vs main empty) · `e4` |
| the catalog has exactly one associate gear (NVFP4) and no W4A16 entry | high | test `tests/test_catalog.py::test_exactly_one_associate_gear` · `e5` |
| the Orin card renders `GATEWAY_READ_TIMEOUT=7200` and other cards do not | high | `tests/test_orin_gateway_read_timeout.py` · file `lobes/profiles/builtin/orin.toml` · `e6` |
| the `orin-associate` shape renders associate at 1048576 / 0.70 / 8192 / max_num_seqs 2 with embedder and reranker hosted | high | `tests/test_orin_associate_shape.py` · goldens `tests/goldens/shapes/orin-associate__orin.env` · `e8`, `e10` |
| on the physical Orin the rendered shape boots at 1M, advertises context 1048576, and passes probes, cold ≥1M needles (streamed and non-streamed) through the gateway and a 2-session agentic run, with associate restarts 0 and no OOM | high | file `docs/evidence/2026-09-13-accept-orin-associate-1m.txt` · `e7`, `e9`, `e11`, `e12` |
| NVFP4 is the A/B winner under the c24 rule | high | file `docs/evidence/2026-09-13-measure-associate-budget-orin-1m.txt` (verdict section) · `e14` |
| docs are reconciled to the 1M budget, cite both transcripts, and keep retired figures only as history | medium | grep evidence `e20`, `e21` (context review); `e19` is a filed false fail pending the operator's rejection |
| the shipped shape's operating headroom at 1M is 1,925 MiB minimum available host memory | medium | accept transcript step 12 (30 s samples; cause of the gap to the A/B's 2,588 MiB not isolated) · `e17` |
| agentic sessions through the gateway run as fast as the A/B's direct-lane run | unverified | contradicted, not claimed: 198.4 s through the gateway vs 119.1 s direct, cause not isolated (accept step 11) |

## Remaining Work / Follow-up

- **Merge order** — this branch stacks on PR #258 (Thor evidence). #258 must merge first, or this PR's diff includes its commits. Owner: operator.
- **Operator adjudication of the delivery ledger** — confirm or reject `o1`–`o15`, `e1`–`e21` and `b1`–`b4` (recommend rejecting `e19`).
- **Open frame parks:**
  - `v3` — head-of-line behaviour at `max_num_seqs=2` behind a cold 1M prefill is unmeasured.
  - `v4` — NVFP4 prefix-cache retention across several long contexts is unmeasured.
  - `v5` — has evidence now (accept step 10: no bytes before the first token) but stays open until the operator resolves it.
- **Unresolved plan risks** — `r1` (re-scaffold overwrite; nothing operator-kept was lost) and `r3` (acceptance downtime: associate offline 06:00:31–06:06:36Z, gears until 06:09:40Z) can be resolved by the operator from the accept transcript.
- **Agentic slowdown through the gateway** — 198.4 s against the A/B's 119.1 s direct, with TTFT p50 15.9 s against 8.6 s. The cause is not isolated. Next step: rerun the 2-session agentic benchmark directly on the lane and through the gateway on the same boot.
- **Headroom gap** — the minimum available memory was 1,925 MiB on the shipped shape against 2,588 MiB in the A/B. The cause is not isolated. Watch it before adding anything resident (`hand` stays out; see issue #216).
- **Orin reranker scoring change** (delta `b4`) — the #227 judge-prompt template is now live on the Orin. Check any Orin-local rerank consumers for threshold assumptions.
- **Mesh members and cold 1M requests** — per decision c30, the Spark, Thor and gateway-only members keep a 600 s read timeout, so cold 1M associate requests are supported only via the Orin gateway. A mesh-wide change would be a separate decision.
- **Orin housekeeping** — the scoped sudoers rule `/etc/sudoers.d/lobes-spike` is still installed (remove with `sudo rm /etc/sudoers.d/lobes-spike` when no longer needed). Backups `~/.lobes/.env.bak-20260913T060018Z-pre-associate-1m` and `~/lobes-dir-backup-20260913T060018Z.tgz` are retained. The Orin CLI is now the branch build 0.77.2, and `MODEL_GEAR_VERSION=0.77.2` is in `.env` until a published release replaces it.
