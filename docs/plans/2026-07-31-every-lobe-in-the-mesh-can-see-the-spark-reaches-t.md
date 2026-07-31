# Build Plan — Every lobe in the mesh can see: the Spark reaches Thor's multimodal worker through its own gateway, and the Qwen cortex itself gains image and video intake

slug: `every-lobe-in-the-mesh-can-see-the-spark-reaches-t` · status: `exported` · from frame: `every-lobe-in-the-mesh-can-see-the-spark-reaches-t`

> Every lobe in the mesh can see: the Spark reaches Thor's multimodal worker through its own gateway, and the Qwen cortex itself gains image and video intake

## Tasks

### t1 — Falsify the boundary claim: prove the worker proxy needs NO change inside lobes/

- covers: c6, h9
- acceptance:
  - gateway/`_config.py` at 0.54.7 is shown to carry worker in all three of `PEER_ORIGIN_ENV`, `PEER_PROXY_ENV` and `PEER_API_KEY_ENV`
  - the packaged fleet template lobes/templates/fleet/docker-compose.yml is shown to pass `WORKER_BASE_URL`, `WORKER_FEASIBLE`, `WORKER_PEER_ORIGIN`, `WORKER_PEER_PROXY` and `WORKER_PEER_API_KEY` into the gateway service
  - a written verdict states either 'no lobes/ change needed -- halves are independent' or names the exact file that must change, which would invalidate boundary claim c6

### t2 — Capture the live before-state baseline from the Spark and the Thor

- covers: c4, h7
- acceptance:
  - Thor GET /capabilities output is captured showing muse feasible=false `hosted_by`=null AND worker feasible=true ready=true
  - the Spark's ~/.lobes/.env is shown carrying `MUSE_PEER_ORIGIN` pointed at Thor and `MODEL_GEAR_VERSION`=0.52.3
  - the Spark's deployed docker-compose.yml is shown containing `MUSE_`\* gateway passthrough lines and zero `WORKER_`\* lines
  - a model=worker request against the Spark is captured failing BEFORE any change, establishing the baseline

### t3 — Audit the role-contract cost claim and how callers address the fleet

- covers: c2, c5, h5, h8
- acceptance:
  - lobes capabilities output is shown listing `final_decision`/`repo_action` in `forbidden_responsibilities` for senses and `final_decision`/`security_decision` for worker
  - the culture/colleague backend config and eidetic's embed URL are each shown pointing at the Spark gateway and addressing roles by name, not by raw model id
  - if any named consumer turns out to dial Thor directly or use a raw model id, that is written down as a correction to audience claim c2

### t4 — Re-scaffold the Spark deployment: packaged fleet compose + version re-pin

- depends on: t1
- acceptance:
  - ~/.lobes/docker-compose.yml is replaced by the packaged lobes/templates/fleet/docker-compose.yml from >=0.54.7, and a diff of what the old hand-patched copy carried (`GATEWAY_API_KEY`, `CULTURE_VLLM_API_KEY`, `MULTIMODAL_PEER_PROXY`, `MULTIMODAL_PEER_API_KEY`) confirms nothing needed is lost
  - `MODEL_GEAR_VERSION` in ~/.lobes/.env is re-pinned to >=0.54.7 so the gateway image installs a lobes-cli that knows the worker role
  - the previous deployment is preserved as a timestamped backup before any file is overwritten

### t5 — Replace the Spark's dead muse peer block with a live worker peer block

- depends on: t4
- covers: c9, h2
- acceptance:
  - `MUSE_PEER_ORIGIN`, `MUSE_PEER_PROXY` and `MUSE_PEER_API_KEY` are removed from ~/.lobes/.env and `MUSE_FEASIBLE` is left unset
  - `WORKER_PEER_ORIGIN`=<http://thor.tail0be7e0.ts.net:8000> and `WORKER_PEER_PROXY`=true are set, with `WORKER_PEER_API_KEY` left EMPTY because Thor declares no inbound gate
  - after bring-up, GET /capabilities on the Spark reports muse with `hosted_by`=null and a model=muse request returns 404 `role_infeasible` -- NOT a proxied 503 `backend_unavailable`
  - lobes overview --list still lists the muse catalog entry and lobes init --shape thor-muse still renders, proving cite-don't-delete held

### t6 — Bring the Spark gateway up and prove the worker relay end to end

- depends on: t5
- acceptance:
  - the rebuild is run with BOTH -f docker-compose.yml and -f docker-compose.shape.yml, and vllm-multimodal is confirmed NOT to have started (the dropped lobe must not eat cortex's reclaimed budget)
  - POST /v1/chat/completions with model=worker returns 200 and the response carries X-Lobes-Proxied-By naming the Thor origin
  - the same request with an `image_url` content part returns a description matching the image, and a negative control image does not match
  - GET /capabilities shows worker feasible=false proxied=true `hosted_by`=<thor origin>, and /v1/models lists the proxied worker
  - the live cortex on the Spark is confirmed still serving after the gateway rebuild -- no regression to the box's primary duty

### t7 — Write the worker-proxy evidence transcript with an explicit split verdict

- depends on: t6
- covers: c8, h10
- acceptance:
  - docs/evidence/2026-XX-XX-accept-worker-proxy-spark.txt exists following the accept-\* naming convention
  - it opens with a VERDICT block naming what it proved (relay, attribution, muse honesty) AND what it did not prove (nothing about cortex vision, nothing about the cortex swap)
  - every command and its raw output are included, not summarised

### t8 — Add unsloth/Qwen3.6-27B-NVFP4 to the catalog as an untested candidate

- depends on: t1
- acceptance:
  - lobes/catalog.py gains a SupportedModel with `role_hint`='candidate', status='untested', `native_max_model_len`=262144, quantization='compressed-tensors', `tool_parser`='`qwen3_coder`'
  - the entry's comment records what was verified from the published checkpoint (`language_model_only`=false, ViT depth 27, `image_token_id` AND `video_token_id`, 15 real mtp.\* tensors, 23.42 GB, `preserve_thinking` in its own chat template) and states plainly that NONE of it is a boot
  - the primary compose lane is left BYTE-UNCHANGED -- this task ships no default change
  - a docs/ page for the checkpoint exists and declares the model UNVALIDATED with no `gpu_mem_util` or `max_model_len` number in it
  - existing catalog tests still pass, including the assertion that `tool_parser` equals `infer_parser`

### t9 — Boot the multimodal cortex candidate on the GB10 and MEASURE its budget

- depends on: t8, t7
- acceptance:
  - a hand-tuned lane serving unsloth/Qwen3.6-27B-NVFP4 boots healthy on the Spark with --quantization=compressed-tensors, no --language-model-only and no --tokenizer override
  - the real `gpu_mem_util` and `max_model_len` that BOOT are recorded from the live run -- refused values are recorded too, exactly as thor-muse's refused 0.40 and thor-worker's accepted 0.45 were
  - the vLLM log lines for weight load size, KV-cache pool and concurrency multiplier are captured verbatim
  - whether embedder, reranker and embed-deep still co-reside at the new budget is stated explicitly
  - if the checkpoint refuses to boot, that is recorded as the outcome and assumption c13 is marked disproven rather than retried indefinitely

### t10 — Probe image AND video intake on the cortex candidate with negative controls

- depends on: t9
- covers: c3, h6
- acceptance:
  - model=cortex with an `image_url` content part returns a description matching the image, and a different control image does not match
  - model=cortex with a video clip returns a description of subject, scene and motion, matching the same standard the worker acceptance used
  - MTP draft acceptance rate is read from the live log and recorded, resolving the parked unknown about whether the self-hosted MTP head engages

### t11 — Prove the swap preserves `preserve_thinking` and strict tool calling

- depends on: t9
- covers: c10, h3
- acceptance:
  - a two-turn prompt-token-count delta shows historical <think> blocks are still retained, matching issue #93's diagnostic
  - a tool call with strict:true and thinking ENABLED returns a clean structured tool call -- not a 500 grammar rejection and not a mangled salvaged name
  - the lane is confirmed to still carry --tool-parser-plugin, --reasoning-parser=qwen3 and --default-chat-template-kwargs `preserve_thinking`, with `GATEWAY_FORCE_STRICT_TOOLS`=1 armed
  - if either probe fails, the promotion is blocked and the failure is recorded rather than worked around

### t13 — Prove both halves in one session and decide the cortex promotion

- depends on: t7, t10, t11
- covers: c1, h1
- acceptance:
  - in a SINGLE session against the live Spark, model=worker with an image returns a correct answer relayed from Thor, and model=cortex with an image returns a correct answer served locally -- each with a negative control
  - a second evidence transcript covering the cortex probes lands under docs/evidence/ with its own split VERDICT block
  - an explicit promote-or-hold decision is recorded: promote unsloth/Qwen3.6-27B-NVFP4 to `role_hint`='primary' and rewrite the compose lane, or hold it at candidate with the reason stated
  - if promoted, the measured budget from t9 is what ships -- no computed or hypothesised number enters the repo

### t12 — Rewrite the now-false rationales in CLAUDE.md and the catalog

- depends on: t10, t11, t13
- acceptance:
  - mmangkad/Qwen3.6-27B-NVFP4's recorded rationale no longer claims it is the tokenizer source or the only vision-capable 27B; the entry itself stays (cite-don't-delete)
  - CLAUDE.md's cortex paragraph stops describing the served model as text-only with the ViT removed, once the promotion lands
  - CLAUDE.md's already-stale worker text is corrected: it describes the thor-worker shape and its compose service as forthcoming with no measured budget, when the shape has landed with MEASURED values (util 0.45 / 262144, KV 41.78 GiB, 14.07x) and an evidence transcript
  - the muse paragraph is updated to say no box declares `MUSE_PEER_ORIGIN` anywhere in the mesh, now that the Spark's dead referral is gone

## Risks

- [unknown_nonblocking] Booting a new cortex on the Spark takes down the mesh's LIVE reasoning lobe. The Spark is the box that colleague, eidetic and the lobes agent itself depend on for model=main/cortex, and a 27B reload is minutes, not seconds. t9 must schedule the boot deliberately and keep a byte-for-byte rollback to the current sakamakismile lane. (task t9)
- [unknown_blocking] If t1 falsifies the boundary claim -- i.e. the worker proxy DOES need a change inside lobes/ -- then c6 is wrong, the halves are coupled, and the worker-proxy-first sequencing (c17) must be revisited before t4 starts. (task t1)
- [unknown_nonblocking] Whether the unsloth self-hosted MTP head actually engages on this 27B: `num_speculative_tokens` value and real draft-acceptance rate. The 35B twin hit 89.1% with method=mtp and 2 tokens and the config carries an `unsloth_fixed_mtp` flag, but this checkpoint is unmeasured. Resolved by t10. (task t10)
- [follow_up] A deployed compose that predates a role's env passthrough silently swallows that role's knobs -- this exact trap has now cost time twice, at muse go-live and again here. Candidate home for a fix: lobes doctor / doctor --fix detecting a stale deployed compose.
