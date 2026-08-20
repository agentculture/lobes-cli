# Delivery Summary — nemotron-lightning-worker

plan: `nemotron-lightning-worker` · run: `partial` · date: `2026-08-20`
baseline: `devague summary skeleton`

## Intent

Execute the converged `nemotron-lightning-worker` plan (PR #189, spec PR #188;
issues #187/#186/#183): move the Thor `worker` seat to Nemotron 3.5 Lightning
on the new fleet nightly, repoint the cortex proxies, and validate hand
budgets. Mid-run, live sm_110 evidence invalidated the plan's premise that the
Thor can serve Lightning; deviation `d1` (operator-approved) swapped the
topology — the run therefore delivered the plan's *goals* on a different box
assignment than its tasks assumed. Delivered as PR #190.

## Planned Work

Quoted verbatim from the `devague summary` skeleton (see
`docs/plans/2026-08-20-nemotron-lightning-worker.md` for full acceptance
criteria):

- `t1` — Thor groundwork spike: pull the deployment to merged main, render, pull the 8bd082 nightly digest, and PROVE `sm_110` compatibility (cuobjdump SASS/PTX listing or a minimal engine boot) before anything else consumes the image
- `t2` — Spike: Lightning serves on the nightly on Thor — standalone vllm serve of nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 (`nemotron_h`) with --reasoning-parser `nemotron_v3` + candidate tool parser; probe structured `tool_calls`, measure plain decode, then evaluate MTP/DSpark separately; grow `max_model_len` progressively from a modest window
- `t3` — catalog.py: add the Lightning entry (`role_hint`=worker, quantization=modelopt, NemotronHForCausalLM facts from the fetched config) and demote unsloth/Qwen3.6-35B-A3B-NVFP4 to a kept candidate (cite-don't-delete); tests updated
- `t4` — roles.py: redefine `ROLE_RESPONSIBILITIES`['worker'] per #187 — drop `image_understanding`/`video_understanding`, keep `tool_use`/execution/`ground_work`, add explicit tokens splitting repo inspection/navigation/run-authorized-commands from code authoring (extend the vocabulary, never prose or model-name checks); adjust cortex's only-role-that-sees-and-decides comment; tests updated
- `t5` — Worker served-id rollout audit + rollout notes: grep sibling checkouts (colleague, embodiment, eidetic, reachy-mini-cli) for unsloth/Qwen3.6-35B-A3B-NVFP4, write the worker rollout-notes doc naming every pinner, and list every box needing a `WORKER_SERVED_NAME` mirror
- `t6` — Incumbent baseline (playbook §1): benchmark the CURRENT Qwen 35B worker on the NEW 8bd082 engine on Thor before it is gone — this number is unrecoverable after the flip
- `t7` — Rollback readiness: verify Thor disk headroom for ~20 GiB of Lightning weights alongside the kept Qwen checkpoint + image; exercise one dry-run restore to the Qwen shape (or record an explicit operator waiver)
- `t8` — Thor live boot + measurement: flip the worker lane to Lightning (`drop_caches` BEFORE recreate; watch for the orphaned-dependent compose state), measure `gpu_mem_util`/`max_model_len` co-resident with hand+embedder+reranker+audio, run the structured tool-call probe against the served lane, and verify the advert (capabilities) tells the deployed truth — text-only, probe-derived tools, loaded/ready/feasible separate
- `t9` — Commit the measured shape: thor-worker.toml swaps overrides.worker to Lightning with the t8-measured budgets (transcript-cited), env.example worker block updated (quantization=modelopt, parser flags, new served id), compose worker lane flags aligned; shape goldens regenerated
- `t10` — #186 Thor: mirror `PRIMARY_SERVED_NAME`=unsloth/Qwen3.8-27B-NVFP4 beside the peer knobs in the Thor .env, recreate the gateway with the full -f overlay set, and prove cortex by proxy
- `t11` — #186 Orin: same mirror + gateway recreate + proof on the Orin box
- `t12` — hand on Thor (#183/#181): re-run the Orin-template probe on the new nightly — budget at the card window, known-answer completion, structured `tool_calls` via the lfm2 parser — and re-attribute #181's LoRA embedding-slot failure (fixed, changed, or re-blocked)
- `t13` — hand on Spark (#183): first exercise of the declared 0.06 hypothesis with the Orin probe template; adapter serving recorded as blocked on unsloth-cli#16
- `t14` — Validation matrix (#187): run the matched worker-shaped tasks + explicit negative/escalation tasks against Lightning on Thor and compare with the t6 incumbent baseline; record the go/no-go
- `t15` — Docs follow the shipped state: new docs/nemotron-3.5-lightning-30b-a3b-nvfp4.md (fetched-config provenance + measured numbers), demotion notes in docs/qwen3.6-35b-a3b-nvfp4.md, role tables in docs/colleague-stack.md + CLAUDE.md, cognitive-split rationale quoted from #187; repo-wide grep proves no surface still claims worker vision
- `t16` — Final acceptance audit: all four success-signal transcripts landed; after-state clauses each mapped to one; zero diffs under lobes/gateway/ and roles.py channel tables (or the boundary reopened explicitly); remaining gaps stated declared/UNVALIDATED

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | sm_110 proven at wheel + silicon level in the 8bd082 image; deployment CLI 0.50.0→0.57.1; before-state corrections recorded (`docs/evidence/2026-08-20-spike-nightly-sm110-thor.txt`). The `~/.lobes` re-render was deliberately deferred to the flip (recorded in the transcript as sequencing) and later executed as the d1 spark-lobe render. |
| `t2` | delivered (negative) | The spike ran and returned a decisive NO-GO: Mamba-2 SSD warmup wedges on Thor on the fleet nightly, upstream v0.27.1, +flashinfer flag, AND the full Jetson AI Lab flag set (`docs/evidence/2026-08-20-spike-lightning-thor-no-go.txt`). Parser availability confirmed; the serve-on-Thor acceptance is unmeetable on these images — this finding triggered `d1`. |
| `t3` | delivered | Catalog entry (later updated to `load-tested` on Spark evidence) + Qwen 35B demotion; merged commit `66803ba`. |
| `t4` | delivered | Worker vocabulary redefined (vision tokens out; `repo_inspection`, `run_authorized_commands` etc. in; `code_authoring` forbidden); commit `bf9748b`. |
| `t5` | delivered | Audit via GitHub code search (no sibling checkouts on the box — recorded): NO external raw-id pinner exists; `docs/worker-lightning-rollout-notes.md` shipped, later amended for d1. |
| `t6` | delivered (amended) | Baseline captured at 61.2 tok/s — on the PRODUCTION engine, not the new one: the new-engine attempt died in the GDN-MTP kernel (recorded as the load-bearing prelude in `docs/evidence/2026-08-20-baseline-worker-qwen35b-thor.txt`); playbook §1 in fact prescribes the production engine. |
| `t7` | partial | Disk headroom verified (247 GiB), byte-identical deployment snapshots taken on Thor AND Spark, old image + checkpoint retained. The dry-run restore was NOT exercised and no explicit waiver was recorded — remaining work. |
| `t8` (`d1`) | delivered as deviated | The live boot + measurement + tool probe + advert verification happened on the SPARK per d1: 17.85 GiB weights, KV 3,560,789 tokens (54.33× @65K), 75.1 tok/s, ~75 ms TTFT, `nemotron_v3`+`qwen3_coder` validated, advert truthful (`docs/evidence/2026-08-20-accept-worker-hand-spark.txt`). |
| `t9` (`d1`) | partial | env.example + compose worker flags landed (`WORKER_REASONING_PARSER`, `WORKER_SPECULATIVE_CONFIG`, quantization docs) and goldens regenerated — but `thor-worker.toml`'s override still declares the Qwen 35B: the shape no longer describes any deployed box (the Spark carries Lightning via `.env`), and writing Spark-measured budgets into a `thor-*` shape needs the shape-naming follow-up first. |
| `t10` | delivered (superseded) | Thor answered `model=cortex` 200 by proxy under the new id (`docs/evidence/2026-08-20-accept-cortex-repoint-thor.txt`) — then d1 made Thor the cortex HOST; the transcript stands as #186 acceptance, the topology superseded same-day (`docs/evidence/2026-08-20-accept-cortex-local-thor.txt`). |
| `t11` | delivered (superseded) | Orin proved 200-by-proxy under the new id (`docs/evidence/2026-08-20-accept-cortex-repoint-orin.txt`), then was repointed at the Thor origin per d1 and re-proven (`ORIN-VIA-THOR-OK`). |
| `t12` | delivered (negative) | hand on Thor stays blocked — but #181 is RE-ATTRIBUTED: the boot-time LoRA failure is gone; the real blocker is sm_110 inference corruption for the LFM2 conv path on BOTH engines, with the Spark as clean control (`docs/evidence/2026-08-20-hand-thor-blocked-reattributed.txt`). |
| `t13` | delivered (exceeded) | Spark 0.06 budget VALIDATED (within 0.5% of Orin) with passing probes — first engine-direct (`docs/evidence/2026-08-20-accept-hand-spark.txt`), then GATEWAY-FRONTED after the d1 re-scaffold, which upgrades #183 item 4 beyond its Orin-era status. |
| `t14` (`d1`) | partial | Matched performance probes ran (decode, TTFT, known-answer, structured tool-calls; Lightning 75.1 tok/s / 75 ms vs incumbent 61.2 tok/s / multi-second turns) with the different-box caveat recorded. The full #187 worker-shaped task list and the negative/escalation tasks were NOT run — remaining work. |
| `t15` | delivered | Six-file docs sweep (commit `ccd86f8`) + per-model Lightning doc + demotion notes + machine-profiles sm_110 lesson; vision-claims grep clean. |
| `t16` | delivered | This artifact + the PR #190 body are the audit: tri-box capabilities sweep recorded (all three boxes resolve every hosted/referred role), evidence transcripts mapped per claim below, and the c7 zero-gateway-diff boundary was REOPENED EXPLICITLY (the hand-proxy reversal), which is the boundary's own recorded escape clause (h5). |

## Mid-work Decisions

- `d1` — TOPOLOGY SWAP: Thor stops hosting worker and instead serves cortex locally (unsloth/Qwen3.8-27B-NVFP4, dense sm_110-safe arch); the Spark loses cortex and gains worker (Lightning) + hand (LFM2.5) — both validated/expected-good on sm_121; Orin and Spark repoint cortex at the Thor origin; freed Spark memory is earmarked for hand adapter fine-tuning (unsloth-cli#16 path) — *reason (from the record): live sm_110 evidence invalidated the plan premise that Thor hosts Mamba/conv-hybrid gears on the 8bd082 nightly; dense-transformer paths serve fine on Thor.* (The "dense" wording in the record predates the discovery that the Qwen line is GDN-hybrid; the operative split is GDN-non-MTP-works / MTP-and-Mamba-and-conv-don't.)
- Thor cortex runs with **MTP OFF** — not in the plan: the sm_110 GDN-**MTP** kernel is absent, the non-MTP path works; the lane's hardcoded flag was hand-removed on the box and the `PRIMARY_SPECULATIVE_CONFIG` off-switch was shipped upstream the same day.
- The `NEVER_PROXIED_BACKENDS = {"hand"}` design decision was **reversed** (not silently — the constant survives empty as the declared-exemption guard): d1's evidence falsified "small enough to run everywhere ⇒ serves everywhere"; hand joined the peer channels AND server resolution tables together per the 0.54.6 lesson.
- The incumbent baseline moved from "new engine" to "production engine" after the new-engine attempt crashed — which playbook §1 in fact prescribes.
- Operator decisions in-session: Qwen3.8-as-local-cortex named as the Lightning fallback before t2 concluded; the swap chosen over keeping Spark-cortex ("spark loses cortex, wins worker and hand"); cortex downtime accepted for two Thor Lightning test windows; a NVIDIA-shareable repro branch requested and shipped.
- The Spark re-scaffold silently reset `VLLM_PORT` (8001→8000) and dropped the inbound `GATEWAY_API_KEY`; both were caught and restored during bring-up (the #92 operator-knob re-render gap, again).

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|-----------------------|----------------|
| `t2` (`d1`) | Lightning cannot serve on Thor on these images (Mamba-2 SSD warmup wedge; four attempts incl. the verbatim Jetson AI Lab recipe) | needs-follow-up |
| `t8` (`d1`) | live sm_110 evidence invalidated the plan premise that Thor hosts Mamba/conv-hybrid gears on the 8bd082 nightly — executed on the Spark instead | needs-follow-up |
| `t9` (`d1`) | the shape whose override the task would swap no longer describes any deployed box; Spark carries the config via `.env` pending the shape-naming follow-up | needs-follow-up |
| `t14` (`d1`) | matched perf probes ran cross-box (the deployed comparison); the full #187 task matrix + negative/escalation tasks not yet run | needs-follow-up |
| `t6` | baseline taken on the production engine after the new-engine attempt crashed in the GDN-MTP kernel; playbook §1 prescribes the production engine anyway | acceptable |
| `t7` | restore dry-run not exercised, no waiver recorded — snapshots + retained artifacts exist | needs-follow-up |
| `t10`/`t11` | delivered to contract, topology superseded same-day by d1 (Orin re-pointed at Thor; Thor serves locally) | acceptable |
| `t16` | the c7 "zero gateway diffs" boundary was exceeded via its own explicit-reopen clause (hand-proxy reversal, commit `f6b8783`-adjacent) | acceptable |

## Evidence

- tests: `uv run pytest -n auto` — **2948 passed, 15 skipped** at `7b6ce74` (post-merge of all agent branches); lint (black/isort/flake8) clean
- commits: `ae61977..7b6ce74` (26 commits on `feat/nemotron-lightning-worker`)
- PRs: #188 (spec, merged) · #189 (plan, merged) · #190 (this delivery, open — gate 3)
- transcripts (all committed): `docs/evidence/2026-08-20-spike-nightly-sm110-thor.txt`, `-spike-lightning-thor-no-go.txt`, `-baseline-worker-qwen35b-thor.txt`, `-accept-cortex-repoint-thor.txt`, `-accept-cortex-repoint-orin.txt`, `-accept-cortex-local-thor.txt`, `-accept-hand-spark.txt`, `-hand-thor-blocked-reattributed.txt`, `-accept-worker-hand-spark.txt`
- deviation ledger: `.devague/deliveries/nemotron-lightning-worker.json` (`d1`, approved)
- repro for NVIDIA: branch `repro/nemotron-lightning-thor-sm110` (pushed)
- live smoke at close: Thor `model=cortex` → "OK"; tri-box capabilities sweep all-green for hosted/referred roles

## Delivery Claims

| Claim | Confidence | Evidence |
|-------|------------|----------|
| The mesh serves the #187 cognitive split: worker=Lightning (Spark), cortex=Qwen3.8 (Thor, 1M, MTP off), senses (Orin) — every box resolves every role | high | the nine transcripts above · tri-box sweep in PR #190 body |
| Lightning outperforms the incumbent worker for callers: 75.1 vs 61.2 tok/s, ~75 ms vs multi-second short turns (different boxes — the deployed comparison, caveat recorded) | high | `docs/evidence/2026-08-20-accept-worker-hand-spark.txt` (+TTFT addendum) |
| Lightning cannot serve on Thor sm_110 on the fleet nightly or upstream v0.27.1, incl. the published Jetson recipe | high | `docs/evidence/2026-08-20-spike-lightning-thor-no-go.txt` · branch `repro/nemotron-lightning-thor-sm110` |
| #186 acceptance met on both boxes (later superseded by d1's better topology) | high | the two repoint transcripts |
| #181 re-attributed: boot-time LoRA failure gone; sm_110 LFM2 inference corruption is the real blocker | high | `docs/evidence/2026-08-20-hand-thor-blocked-reattributed.txt` |
| hand budgets reproducible on Orin AND Spark; Spark now gateway-fronted | high | `docs/evidence/2026-08-20-accept-hand-spark.txt` · `-accept-worker-hand-spark.txt` |
| repo surfaces (catalog, roles, templates, gateway channels, docs, goldens) match the deployed truth | high | commits `ae61977..7b6ce74` · suite 2948 green |
| Lightning MTP/DSpark benefit on the Spark | unverified | not yet evaluated — not claimed |
| strict-tools (xgrammar) behaviour on the Lightning lane | unverified | not yet probed — not claimed |
| rollback path exercised | unverified | snapshots exist; restore never dry-run |

## Remaining Work / Follow-up

- `t14` — run the full #187 worker-shaped + negative/escalation task matrix against the Spark Lightning lane (baseline comparison already recorded).
- `t9` — shape-naming follow-up: shapes proved card-agnostic (thor-worker renders on the Spark; spark-lobe on the Thor); rename/alias, then write the Spark-measured Lightning budgets into the worker-hosting shape.
- `t7` — exercise one restore dry-run from the Thor snapshot (or record an explicit operator waiver).
- Lightning-on-Thor: hand the repro branch to the Jetson AI Lab maintainer (operator has the contact); the open lead is a JetPack/driver/env diff on the same image. If a fix lands, Thor could re-host worker and re-enable cortex MTP (both gated on sm_110 kernels).
- Thor cortex speed: 12.1 tok/s without MTP is the known cost of the swap; re-enable via `PRIMARY_SPECULATIVE_CONFIG` the moment an image ships the sm_110 GDN-MTP kernel.
- Release + deployment refresh: the hand-proxy reversal and the new knobs are in-tree only until 0.58.0 publishes; the Thor gateway then needs a rebuild for its hand referral to annotate (currently an honest 404).
- Spark audio overlay adverts (`stt`/`tts` show not-ready) were out of scope today — verify after the re-scaffold settles.
- hand adapter serving: still blocked on `unsloth-cli#16`; the freed Spark memory is the intended fine-tuning ground (the d1 dividend).
