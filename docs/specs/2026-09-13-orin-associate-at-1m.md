# orin associate at 1M

> The Jetson AGX Orin's associate lane serves Nemotron 3.5 Lightning at the native 1,048,576-token window as shipped, measured lobes configuration — not hand-typed .env keys — using the checkpoint that wins a measured NVFP4-vs-W4A16 A/B on the Orin

## Audience

- the lobes operator who runs the Jetson AGX Orin, and every mesh caller that addresses model=associate (e.g. pi/Qwen-Code agent harnesses via the Orin gateway or a mesh member)

## Before → After

- Before: the Orin associate lane serves Lightning NVFP4 at 128000 tokens on vLLM v0.27.1 with DSpark, util 0.70 and 8192 batched tokens set by hand-typed .env keys; the orin-associate shape still says util 0.80 at 128000, the card doc block says 0.56, and no transcript has exercised the lane past 128000 on the Orin
- After: rendering orin-associate on the Orin produces a lane that boots at `max_model_len` 1048576 with the A/B-measured checkpoint, util, batched-token cap and `max_num_seqs`; the gateway advertises context 1048576; and every value cites a docs/evidence transcript

## Why it matters

- Lightning's 1M native window is the reason to run it on a 64 GB board, and the Thor spike showed recall and agentic tool use hold to 1,040,073 tokens; today the Orin advertises and serves only 128K, and its live budget exists only in hand-edited .env keys that drift from the shipped shape

## Requirements

- the vllm-associate lane gains an `ASSOCIATE_MAX_NUM_SEQS` knob rendered with the worker lane's conditional idiom (unset renders no argv token), so a 1M-window lane can cap concurrent sequences; documented in the lane's knob comment and pinned in tests/`test_associate_compose.py`
  - honesty: docker compose config renders --max-num-seqs=N exactly when `ASSOCIATE_MAX_NUM_SEQS` is set and renders no argv token when unset, asserted in tests/`test_associate_compose.py`
- the vllm-associate lane gains a `VLLM_ALLOW_LONG_MAX_MODEL_LEN` environment passthrough (`ASSOCIATE_ALLOW_LONG_MAX_MODEL_LEN`, default 0) mirroring vllm-primary, required only when the served checkpoint declares `max_position_embeddings` below the requested window
  - honesty: docker compose config renders `VLLM_ALLOW_LONG_MAX_MODEL_LEN`=0 by default and =1 when `ASSOCIATE_ALLOW_LONG_MAX_MODEL_LEN`=1 in the vllm-associate environment, asserted by a test
- orin-associate.toml's associate override declares the measured 1M budget (`max_model_len` 1048576 plus the util, batched-token cap and `max_num_seqs` the A/B measured) with each value citing a docs/evidence transcript, replacing the 128000 window
  - honesty: every numeric override in orin-associate.toml's associate block carries a comment citing the docs/evidence transcript that measured it on the Orin, and the orin-associate golden is regenerated to match
- stale Orin associate statements are reconciled with what ships: the orin-associate header (still describes hand + pooling gears), orin.toml's 0.56 LOCKSTEP doc block, docs/deployment-shapes.md:91 (pre-d2 util 0.63 KV figures) and docs/orin-associate-deployment.md (16384 batched tokens vs the live 8192)
  - honesty: grep for the retired associate budget literals (0.56, 0.63 KV-pool figures, 16384 batched tokens) in the listed docs and profile comments returns only lines explicitly marked historical
- the catalog keeps exactly one `role_hint`="associate" entry (the NVIDIA NVFP4 checkpoint), pinned by a new `test_exactly_one_associate_gear`; the useful-quants W4A16 is NOT added to the catalog (A/B 2026-09-13 missed the c24 bar, v2 resolved)
  - honesty: tests/`test_catalog.py`::`test_exactly_one_associate_gear` fails when a second `role_hint`=associate entry is added and passes on the shipped catalog
  - honesty: lobes/catalog.py contains no useful-quants W4A16 entry and `test_exactly_one_associate_gear` passes on the shipped catalog
- after the window change the Orin gateway advertises context 1048576 for associate on GET /capabilities and lobes capabilities: the gateway container is recreated (not only vllm-associate), and acceptance checks both the advertised context and the live fingerprint `max_model_len` read from the lane's /v1/models
  - honesty: on the live Orin after rollout, curl GET /capabilities through the Orin gateway returns associate context 1048576 AND the replica fingerprint `max_model_len` read from the lane's /v1/models equals 1048576
- every doc that states associate's Orin budget or window as fact is updated to the measured 1M values: CLAUDE.md associate paragraph, docs/deployment-shapes.md orin-associate rows, docs/nemotron-3.5-lightning-30b-a3b-nvfp4.md Orin section, docs/machine-profiles.md totals, docs/orin-associate-deployment.md
  - honesty: each listed doc cites the 1M measure or accept transcript by path where it states the Orin associate budget
- the 1M budget and lane are backed by two new transcripts under docs/evidence/ following the repo naming convention: a measure-associate-budget-orin-1m transcript (the A/B) and an accept-orin-associate-1m transcript of the shipped rendered shape on the physical Orin; until the accept transcript lands the shape stays DECLARED/UNVALIDATED (#108)
  - honesty: both transcripts exist under docs/evidence with the YYYY-MM-DD-measure-/accept- naming and every number in the success signals appears in them with a timestamp or count
- orin-associate.toml's hosts list is associate, embedder and reranker (the pooling gears return to the shape), its header comment describes exactly that, and tests/`test_orin_associate_shape.py` plus the orin-associate golden assert the three hosted roles and the dropped ones
  - honesty: lobes init --shape orin-associate (dry-run) on the orin card renders `ASSOCIATE_`\* plus `EMBED_`\* and `RERANK_`\* lanes and marks cortex/senses/hand/muse/worker infeasible; the regenerated golden and `test_orin_associate_shape` assert exactly that set
- a cold 1M request through the Orin gateway as model=associate completes: the Orin gateway's upstream read timeout (`GATEWAY_READ_TIMEOUT`, template default 600 s, live Orin value 600 s, applied as the socket read timeout on every upstream POST at lobes/gateway/server.py:616) is raised above the measured cold 1.04M TTFT (2,390 s NVFP4 on 2026-09-13) with margin, set through a shipped knob (env.example + orin card or shape), not a hand-typed key
  - honesty: a streamed AND a non-streamed request of >= 1,000,000 prompt tokens sent cold through the Orin gateway as model=associate return 200 with the needle recalled, and the rendered gateway env shows `GATEWAY_READ_TIMEOUT` above the measured cold TTFT, both in the accept transcript
- the accept transcript records embed and reranker restart counts after rollout, and the Orin deployment doc names the observed gear first-start race: on 2026-09-13 model-gear-vllm-embed failed its first start every time the gears started together after associate (02:29, 03:34, 04:31 UTC; 'No available memory for the cache blocks') and recovered via restart=unless-stopped
  - honesty: the accept transcript quotes docker inspect RestartCount for vllm-embed and vllm-rerank after rollout, and docs/orin-associate-deployment.md names the gear first-start KV race with the 2026-09-13 timestamps
- orin-associate.toml keeps the pre-change 128000-window associate values as a documented rollback in its comment block (the spark-lobe d4 idiom), and the rollout records a dated backup of the Orin .env before rendering
  - honesty: orin-associate.toml's comment block quotes the 128000 / util 0.70 / 8192 pair as a rollback, and the accept transcript shows the dated .env backup path taken before render
- the Orin associate docs record the measured zero-swap headroom at 1M (minimum available host memory 2,588 MiB during the NVFP4 1.04M prefill, 2026-09-13) together with the operating rule that any side process on the Orin runs memory-capped
  - honesty: the Orin associate docs quote the 2,588 MiB minimum available memory with its evidence transcript path and state the memory-capped side-process rule

## Honesty conditions

- a docs/evidence accept transcript shows the RENDERED orin-associate shape (not a hand-run docker command) booting on the physical Orin at `max_model_len` 1048576, with the engine argv, KV pool line, a >= 1M-token needle PASS and GET /capabilities context 1048576
- tests/`test_associate_exposure.py` passes unmodified after the lane-knob change, and docker compose config for vllm-associate still shows --host=0.0.0.0 with no ports and no `network_mode`
- neither orin-associate.toml nor orin.toml declares an image or `speculative_config` key for associate after the change, and the orin card comment still documents `ASSOCIATE_IMAGE` + `ASSOCIATE_SPECULATIVE_CONFIG` as the operator-typed matched pair
- the accept transcript exercises the lane both directly on the Orin and through the Orin gateway as model=associate, the path mesh callers use
- the before-state values (128000 window, util 0.70, 8192 batched tokens, v0.27.1, DSpark, shape 0.80, card 0.56) are quoted from the live Orin .env/engine log and the repo files at the plan's start commit
- the Thor 1M results cited are the committed docs/evidence/2026-09-13-spike-lightning-thor-v029.txt figures, and the Orin's pre-change context advert (128000) is captured from GET /capabilities before rollout
- a dry-run lobes init --shape orin-associate on the orin card renders `ASSOCIATE_MAX_MODEL_LEN`=1048576 and the measured util / batched tokens / `max_num_seqs`, every one of which appears in the measure transcript
- each listed check appears as a timestamped line (or count) in the accept transcript produced from the rendered shape on the physical Orin, including restarts==0 read from docker inspect after the agentic run
- the measure transcript contains one row per arm for every listed metric and a verdict paragraph that applies decision c24's rule explicitly (thresholds quoted, numbers compared)

## Success signals

- on the physical Orin with the rendered shape: /health 200; known-answer, multi-step and tool-call probes PASS; a needle at >= 1,000,000 prompt tokens PASSES; GET /capabilities reports associate context == 1048576; zero OOMKilled/engine-dead events across a 2-session 12-turn agentic run; associate restarts == 0 after the run
- the A/B transcript records, for NVFP4 and W4A16 at 1,048,576: boot result, KV pool tokens, decode tok/s at >= 3 depths, cold TTFT at ~1M, concurrency 1/2/4 aggregate tok/s, agentic tool-call validity and recall counts, peak tj and minimum available memory; the winner is chosen by a stated rule, not by feel

## Scope / boundaries

- the associate lane's exposure contract is unchanged: --host=0.0.0.0 paired with no published host port (expose-only behind the gateway bearer gate), as pinned by tests/`test_associate_exposure.py`
- `ASSOCIATE_IMAGE` and `ASSOCIATE_SPECULATIVE_CONFIG` stay operator-typed .env keys (the v0.27.1 image + DSpark matched pair), never declared on card or shape, unless the plan explicitly decides otherwise

## Non-goals

- the Thor, Spark and every non-Orin box are unchanged; the Thor keeps worker; no cortex/senses/worker/muse lane changes
- no fleet engine-pin change: the Orin keeps vllm/vllm-openai:v0.27.1 as the operator-typed `ASSOCIATE_IMAGE` (DSpark needs it); moving to v0.29.0 is out of scope

## Assumptions

- an `ASSOCIATE_DTYPE` knob is NOT required even if the W4A16 candidate wins: its model card states vLLM 0.27.1 auto dtype already resolves to BF16 for that checkpoint (--dtype bfloat16 in the author's profile is explicit-but-optional)
- no schema, render or lock-allowlist change is needed for `max_num_seqs` or `allow_long_max_model_len` on associate: both are already ungated RoleProfile knobs with env suffixes, and the deployment-lock allowlist derives from those tables; only the compose slot, env.example, shape value, tests and goldens change

## Scope exploration

- `s1` — `lobes/templates/fleet/docker-compose.yml:1718-1738 (vllm-associate command) + :1541 (vllm-worker)`: associate command exposes model/served-name/quantization/kv dtype/max-model-len/gpu util/mamba x5/prefix caching/batched tokens/speculative config but NO --max-num-seqs; vllm-worker already renders ${`WORKER_MAX_NUM_SEQS`:+--max-num-seqs=${`WORKER_MAX_NUM_SEQS`}} — the exact precedent
  - seeds: `c2`
- `s2` — `lobes/templates/fleet/docker-compose.yml:1683-1688 (vllm-associate environment) + :68-84 (vllm-primary)`: associate environment carries only `HF_HOME`, `TOKENIZERS_PARALLELISM`, `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`=0, `MG_LOG_DIR`, `MG_LOG_NAME`; vllm-primary already passes `VLLM_ALLOW_LONG_MAX_MODEL_LEN`=${`PRIMARY_ALLOW_LONG_MAX_MODEL_LEN`:-0} for a window beyond the declared 262144 — the precedent; NVIDIA NVFP4 declares 1,048,576 (not needed), the useful-quants W4A16 declares 262144 (needed)
  - seeds: `c3`
- `s3` — `tests/test_associate_exposure.py:98-202`: pins --host=0.0.0.0 together with no `network_mode`/ports publishing on vllm-associate (the c29 tailnet exposure incident); new decode knobs must not touch it and it must stay green
  - seeds: `c4`
- `s4` — `useful-quants W4A16 model card (README Quick start) + compose associate command`: card: 'On the validated vLLM 0.27.1 stack, automatic dtype selection resolves to BF16 for this checkpoint. --dtype bfloat16 is optional'; the associate lane has no --dtype knob today, so none is needed unless the A/B shows auto resolving differently on the Orin
  - seeds: `c5`
- `s5` — `tests/test_associate_compose.py:33-59 (_EIGHT_FLAGS, _DECLARED_KNOBS) + tests/test_orin_associate_shape.py:126-140 + tests/goldens/shapes/orin-associate__*.env`: every associate knob is enumerated and render-asserted via docker compose config; the orin-associate shape's rendered env is asserted and golden-pinned — each new knob and any shape value change requires extending these tuples and regenerating the orin-associate goldens
  - seeds: `c2`, `c3`
- `s6` — `lobes/profiles/render.py:90-157 (ROLE_ENV_PREFIX, _KNOB_ENV_SUFFIX) + lobes/profiles/schema.py:184-193 (KNOB_LANE_ROLES) + lobes/runtime/_lock.py:67-72,128-152`: `max_num_seqs` -> `MAX_NUM_SEQS` (render.py:116) and `allow_long_max_model_len` -> `ALLOW_LONG_MAX_MODEL_LEN` (:124) exist for every role; neither is in `KNOB_LANE_ROLES` so associate may declare them; `lock_keys`() is the prefix x suffix cross product imported from render.py, so the allowlist follows automatically; no weight-dtype knob exists (only `kv_cache_dtype`)
  - seeds: `c6`
- `s7` — `tests/goldens/regen.py + tests/test_profile_goldens.py + tests/test_shape_goldens.py`: goldens (spark/thor/template-defaults and shapes/<shape>`__`<card>.env) regenerate with 'uv run python tests/goldens/regen.py' and are diffed byte-for-byte; any orin-associate value change must be regenerated and reviewed
  - seeds: `c6`
- `s8` — `lobes/profiles/builtin_shapes/orin-associate.toml:45-96`: hosts=\["associate"\] (solo since deviation d2); overrides.associate: model NVFP4 (:60), `gpu_mem_util` 0.80 MEASURED 2026-08-26 (:84), `max_model_len` 128000 'vendor recipe window, NOT the checkpoint 1,048,576 native ceiling' (:89), quantization modelopt (:92), `kv_cache_dtype` bfloat16 'NOT the declared FP8' (:96); no batched-tokens, speculative or image keys; whole shape DECLARED/NOT VALIDATED (#108, :25)
  - seeds: `c7`
- `s9` — `tests/test_orin_associate_shape.py:83-140 + tests/goldens/shapes/orin-associate__orin.env`: `test_declares_the_full_associate_override_matching_the_card_documentation` asserts shape `gpu_mem_util` == 0.80 and `max_model_len` 128000 but only checks the card text contains '0.56'; golden locks `ASSOCIATE_GPU_MEM_UTIL`=0.8 / `ASSOCIATE_MAX_MODEL_LEN`=128000 — both change with the 1M budget
  - seeds: `c7`
- `s10` — `lobes/profiles/builtin/orin.toml:521-567 + docs/orin-associate-deployment.md:74-85`: the card documents `ASSOCIATE_IMAGE`=vllm/vllm-openai:v0.27.1 and `ASSOCIATE_SPECULATIVE_CONFIG` (DSpark) as an operator-typed matched pair, deliberately not rendered by card or shape (dspark needs v0.27.1; the fleet nightly refuses method=dspark)
  - seeds: `c8`
- `s11` — `lobes/profiles/builtin/orin.toml:231,297-324 + docs/deployment-shapes.md:91 + docs/orin-associate-deployment.md:39`: orin.toml \[roles.associate\] is documentation-only (feasible=false, deviation d1) with a LOCKSTEP block claiming `gpu_mem_util` 0.56 (from the gears-first 2026-08-25 measurement) that no longer matches the shape's 0.80; deployment-shapes.md:91 cites pre-d2 0.63 figures; the deployment doc's docker run uses --max-num-batched-tokens 16384 while the live box runs 8192 (the 2026-09-12 resilience change, undocumented in-repo)
  - seeds: `c9`
- `s12` — `deployments/ (variation catalog)`: only jetson-agx-`thor__thor`-worker exists; there is no jetson-agx-orin\* lock or VARIATION.md, so the Orin's hand-typed deployment (v0.27.1 image, DSpark, util 0.70, 8192) is captured nowhere in-repo
  - seeds: `c9`
- `s13` — `lobes/catalog.py:896-1027 (Lightning NVFP4 entry) + :673-754 (unsloth/gemma-4-12B-it-qat-w4a16)`: the NVFP4 entry is the sole `role_hint`=associate (:1005), `native_max_model_len`=1048576 (:1008), quantization=modelopt (:1013), status load-tested; the Gemma QAT W4A16 entry is the compressed-tensors INT4 precedent: `role_hint`=candidate, status=configured until booted, doc gemma-4-12b-qat-w4a16.md, own test block tests/`test_catalog.py`:868-940
  - seeds: `c10`
- `s14` — `tests/test_catalog.py:40-110,819-843 + lobes/catalog.py resolve_tier (:1408-1432) + lobes/gateway/_config.py:86,874-876`: generic invariants: required fields, unique ids, status in {load-tested, configured}, positive `native_max_model_len`, doc exists, `tool_parser`==`infer_parser`; there is NO `test_exactly_one_associate_gear` and `resolve_tier` returns the FIRST `role_hint` match, so a second associate entry would be silently shadowed; `_DEFAULT_ASSOCIATE` = `resolve_tier`('associate').id feeds the `ASSOCIATE_SERVED_NAME` default
  - seeds: `c10`
- `s15` — `lobes/roles.py:445-453,611-700 + lobes/templates/fleet/docker-compose.yml:1898-1913 + lobes/gateway/_replicas.py:196-199,296-318`: associate context resolves from `ASSOCIATE_MAX_MODEL_LEN` in the GATEWAY process env (peer-advertised > own env > catalog native only when feasible); the gateway service passes `ASSOCIATE_MAX_MODEL_LEN`=${`ASSOCIATE_MAX_MODEL_LEN`:-} solely for /capabilities, so a gateway not recreated keeps the old window; Fingerprint.`max_model_len` is LIVE from the lane's /v1/models and is a DISQUALIFYING pool field
  - seeds: `c11`
- `s16` — `CLAUDE.md:426-428 + docs/deployment-shapes.md:54-55,91 + docs/nemotron-3.5-lightning-30b-a3b-nvfp4.md:195-219 + docs/machine-profiles.md:230,381`: each states the 128000-window budget (util 0.56 / 0.63, KV pool 1,524,000 or 1,249,280 tokens, refused 0.70) as fact; all become historical at 1M and must cite the new measurement
  - seeds: `c12`
- `s17` — `docs/evidence/ naming by example + CLAUDE.md:867 (#108 honesty rule)`: names follow YYYY-MM-DD-{spike|accept|measure|baseline}-<slug>.txt (e.g. 2026-08-25-measure-associate-budget-orin.txt, 2026-08-26-accept-orin-associate.txt); the #108 rule forbids calling any value validated without a live transcript
  - seeds: `c13`
- `s18` — `challenge pass / adjacent-systems lens: lobes/gateway/server.py:585-625 + _config.py:1046 + fleet compose :1917-1918`: upstream read timeout is one socket timeout per POST, default 600 s, live Orin 600 s; cold 1.04M TTFT measured 2,390 s so a cold 1M request through the gateway would time out; mesh members forward with their own cfg.`read_timeout` (not read line-by-line for every forward call site)
  - seeds: `c26`
- `s19` — `challenge pass / failure-mode lens: Orin docker logs vllm-embed/vllm-rerank 02:00-04:33 UTC`: embed logged 'No available memory for the cache blocks' on its first start 3 of 3 times the gears started together after associate, recovered by restart=unless-stopped; rerank restarted once at 02:29 without that error
  - seeds: `c27`
- `s20` — `challenge pass / reversibility lens: orin-associate.toml + Orin ~/.lobes/.env`: spec had no rollback path; live .env already has a dated pre-spike backup, shape TOML carries no 128000 rollback note
  - seeds: `c28`
- `s21` — `challenge pass / operations lens: scratchpad orin-ab/free-mem.log (30 s samples)`: min available host memory 2,588 MiB (NVFP4, 1.04M prefill) and 4,182 MiB (W4A16); zero swap; the 01:45 UTC download OOM shows an uncapped side process kills the associate engine
  - seeds: `c29`
- `s22` — `challenge pass / concurrency lens: orin-ab concurrency.jsonl + max_num_seqs 2`: only 512-token prompts measured at N=1/2/4; long-prompt contention and short turns behind a cold 1M prefill unmeasured (parked v3)
- `s23` — `challenge pass / data-flow lens: needle-warm-*.json + boot.log prefix cache hit rate`: NVFP4 dropped 128K/250K prefixes after a 1M request, W4A16 kept them; align-mode prefix caching flagged experimental by vLLM (parked v4)
- `s24` — `challenge pass / hardware lens: orin-ab tegrastats.log`: clean: peak tj 94.0 C NVFP4 / 94.5 C W4A16 through 1M prefill; no throttle events examined beyond tegrastats temps
- `s25` — `challenge pass / security lens: tests/test_associate_exposure.py + c4`: clean: no new port or `network_mode`; the timeout knob and gateway recreate do not touch the expose-only contract; W4A16 not adopted so no new third-party checkpoint enters
- `s26` — `challenge pass / mesh lens: lobes/gateway/_replicas.py Fingerprint.max_model_len + roles.py context`: only the Orin hosts associate, so no pool disagreement from `max_model_len` 1048576; mesh members' cold-request timeout is the open question q1
- `s27` — `challenge pass / unexamined: Spark/Thor/gateway-only member gateway env, vLLM streaming first-byte timing, deployment lock capture`: not examined live this pass: other members' `GATEWAY_READ_TIMEOUT` values, whether a stream emits bytes before the first token (parked v5), and Orin lock capture (s12, out of scope)

## Decisions

- the measured 1M values become authoritative in orin-associate.toml (util, batched-token cap, `max_num_seqs`, window) with evidence citations; the orin card doc block, docs and the live Orin .env are reconciled to match the shape (operator 2026-09-13)
- the shipped orin-associate shape hosts associate + embedder + reranker (matching the live box and the A/B's gears-resident measurement); hand stays out, as in the solo d2 shape (operator 2026-09-13)
- A/B winner rule: adopt useful-quants W4A16 only if it boots at 1,048,576, passes every probe and needle (including >= 1M tokens) and the 2-session agentic run with zero OOM, AND beats NVIDIA NVFP4 by >= 10% on decode at depth or agentic wall time; otherwise associate stays on NVFP4 (operator 2026-09-13)
- cold >600 s-TTFT associate requests are supported via the Orin gateway only: the `GATEWAY_READ_TIMEOUT` raise is set on the Orin, non-Orin mesh members stay unchanged (c20), and the limit plus the opt-in knob are documented (operator 2026-09-13, resolves q1)

## Open parks

- [unknown_nonblocking] head-of-line blocking at `max_num_seqs`=2: one cold 1M prefill (~40 min measured) holds one of two sequence slots; concurrency was measured only with 512-token prompts, so behaviour of a second long request, or of short turns queued behind a cold 1M prefill, is unmeasured
- [unknown_nonblocking] prefix-cache retention across contexts on NVFP4: after a 1.04M request the 128K and 250K prompts were NOT served from cache (TTFT 90.5 s and 175.8 s, same as cold); vLLM flags Mamba align-mode prefix caching experimental; the cost for sessions that alternate between several long contexts is unmeasured
- [unknown_nonblocking] whether vLLM v0.27.1 streaming emits any bytes (role chunk, headers) before the first generated token; if it does, a streamed cold 1M request might survive a 600 s gateway read timeout while a non-streamed one does not; unprobed

## Resolved vagueness

- [unknown_blocking] which source is authoritative for the Orin associate budget: the orin-associate shape (util 0.80, solo), the orin card doc block (util 0.56, LOCKSTEP comment), or the live box (util 0.70, 8192 batched tokens, embed+rerank resident) — shipping 1M as measured config must reconcile all three — resolved: the measured 1M values in orin-associate.toml are authoritative; the orin card doc block, docs and the live .env are reconciled to the shape
- [unknown_blocking] the useful-quants W4A16's catalog `native_max_model_len`: its config.json declares 262144 (repo discipline is config over card) while the card and NVIDIA's NVFP4 claim 1M; only a live 1M needle on the Orin can justify declaring 1048576 — settled by the A/B — resolved: Operator decision 2026-09-13 after the Orin 1M A/B (NVFP4 vs useful-quants W4A16, DSpark x5, bf16 KV, util 0.70, max-num-seqs 2, v0.27.1): W4A16 booted at 1,048,576 and passed every probe, needle (to 1.04M) and the 2-session agentic run with zero OOM, but did not clear the c24 >=10% bar (decode at 1.04M depth -7%: 7.30 vs 7.85 tok/s avg cold+warm; agentic wall -1% 1-session, -0.7% 2-session). KEEP NVFP4 on associate; no W4A16 catalog entry or swap; its 262144 `native_max_model_len` is moot. W4A16's capacity-only edges (+8% KV pool, 128K/250K prefix cache retained after a 1M request) are recorded in the evidence as measured, not adopted.
