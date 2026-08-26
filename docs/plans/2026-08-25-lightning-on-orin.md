# Build Plan — Lightning on Orin

slug: `lightning-on-orin` · status: `exported` · from frame: `lightning-on-orin`

> lobes serves NVIDIA Nemotron 3.5 Lightning 30B-A3B on the Jetson AGX Orin 64GB

## Tasks

### t1 — Capture the board's pre-spike container inventory and restore it

- covers: c33, h14
- acceptance:
  - A file under docs/evidence/ lists the containers running before the first stop (llamacpp-cortex, model-gear-gateway, model-gear-vllm-embed, model-gear-vllm-rerank, prod-worker-1)
  - All four stopped lanes run again and the gateway answers /health, OR the transcript explicitly declares them abandoned and why

### t2 — Finalise the vLLM spike transcript as citable evidence

- covers: c43, c42, c44, c45, c36, c32, c26, h19, h20, h21, h28, h13, h30, h31, h26, c40, h29
- acceptance:
  - The transcript states GO in its own words and carries 19.45 GiB weights, 15.07 GiB KV, 1,720,320-token pool, 13.44x at 128k, known-answer PASS, tool-call PASS, and both decode runs
  - Every memory figure names its instrument (engine boot log vs free vs docker stats vs tegrastats)
  - The decode figure states its speculation setting and is never compared to the vendor's 89 tok/s
  - It records that the Mamba2 SSD Triton warmup appeared AND completed, distinguished from FlashInfer SSU selection
  - The transcript records the third recipe defect: the DSpark repo id lacking its -NVFP4 infix, quoted from the resolved SpeculativeConfig vLLM actually loaded

### t3 — Finalise the llama.cpp spike transcript as citable evidence

- covers: c38, c27, c19, c29, c34, h11, h17, h15, h27, h7
- acceptance:
  - The transcript states NO-GO in its own words, quotes the missing-`Q4_K_M` error and the blk.5.`ssm_in`.weight failure, and names llama.cpp b10373
  - It attributes the failure to a build-version gap, NOT `sm_87`, and records that a newer llama.cpp is untried
  - It names which run was verbatim and which corrected, with the `Q4_K_M` -> `Q4_0` basis stated

### t4 — Carve Lightning out of the W4A4 infeasibility claim on the Orin card

- covers: c2, h1, c25, h25
- acceptance:
  - orin.py and builtin/orin.toml cite Lightning's own `hf_quant_config.json` (`W4A16_NVFP4` experts, FP8 `in_proj`/`out_proj`) as why W4A4 does not apply to it
  - A test asserts Qwen3.8-27B-NVFP4 and Gemma-4-31B-IT-NVFP4 remain declared infeasible on `sm_87`
  - uv run pytest -n auto passes

### t5 — Correct the stale v0.27.1-on-Thor sentence in the Lightning doc

- covers: c7, h3
- acceptance:
  - docs/nemotron-3.5-lightning-30b-a3b-nvfp4.md no longer calls the v0.27.1 Thor spike 'not yet run' and matches catalog.py
  - It cites the 2026-08-25 Orin result and states the Thor wedge is `sm_110`-specific, since the Orin cleared the identical warmup

### t6 — Add the 'associate' role to the role vocabulary and capability ordering

- depends on: t4
- covers: c13, h4, c1, h33, c22, h22, c24, h24
- acceptance:
  - roles.py declares associate: execution/`ground_work`/`bulk_transform`/drafting/`repo_inspection`/`tool_use`; forbidden `final_decision`/`security_decision`/`code_authoring`/`repo_action`
  - catalog.py `TIER_ROLE` places associate as hand < multimodal < worker < muse < associate < main, asserted by a test
  - capabilities, /v1/models, model= aliasing and lobes up all address associate; a test asserts an unhosted associate 404s `role_infeasible`, never a silent downgrade
  - associate sheds 429 under pressure like worker/muse; a test asserts hand remains the only servable floor
  - No existing role's routing, selection, replica-pool or proxy behaviour changes: a test asserts the nine pre-existing role prefixes route identically before and after, and that a deployment declaring no associate config is byte-identical to the pre-associate contract
  - associate gains its <PREFIX>`_PEER_ORIGINS` / `_PEER_API_KEYS` / peer-proxy vocabulary in `_config.py`, `_replicas.py`, `_routing.py` and `_selection.py` exactly as the existing nine have it (#199, 0.63.0)

### t7 — Give the eight unexpressible vLLM serve flags a real home

- covers: c16, h6, c15, h5
- acceptance:
  - The five mamba flags plus --enable-prefix-caching, --max-num-batched-tokens and --trust-remote-code each reach a real flag or are dropped with a recorded reason
  - `ASSOCIATE_IMAGE` overrides the lane image and docker compose config proves it reaches the image field
  - A test asserts every declared knob renders — no dead declarations (#92)

### t8 — Declare associate on the Orin card profile with a measured budget

- depends on: t2
- covers: c4, h2, c20, h8, c23, h23
- acceptance:
  - builtin/orin.toml declares associate feasible with model, `gpu_mem_util` and `max_model_len` measured on THIS board, not the vendor's 0.7 copied forward
  - A refused util is recorded in the profile comment rather than quietly lowered
  - \[\[`exclusive_roles`\]\] names the group associate collides with, its reason quoting measured arithmetic and naming its instrument

### t9 — Ship the orin-associate deployment shape

- depends on: t7, t8
- covers: c21, h10
- acceptance:
  - An orin-associate shape hosts associate + hand + embedder + reranker, drops cortex and senses, and declares no audio
  - lobes init --shape orin-associate renders a byte-stable .env and docker-compose.shape.yml, with goldens committed
  - Re-running with the previous shape restores the deployment byte-for-byte

### t10 — Put the associate lane behind the gateway's authenticated front

- depends on: t9
- covers: c30, c46, h12, h32
- acceptance:
  - The rendered lane is not reachable unauthenticated from the tailnet; a probe from a peer without `GATEWAY_API_KEY` is refused
  - The lane binds through the gateway rather than publishing an open port, and a test asserts the rendered compose exposes no unauthenticated generate port
  - The evidence file quotes the two observed tailnet client IPs as the motivating incident

### t11 — Verify the untouched boundaries by test, not by assertion

- depends on: t6
- covers: c9, h34, c10, h35, h36
- acceptance:
  - A test asserts no file under lobes/gateway/ or the proxy/role mechanism changed relative to main, beyond the associate additions in roles.py/catalog.py
  - The Spark's render and .env are untouched; worker keeps serving and its goldens are unchanged

## Risks

- [unknown_nonblocking] t1 (restore the board) and t8 (measure associate's budget on the board) are in tension: the budget measurement needs the heavy lanes stopped again, so restoring first means a second disruption. Sequence deliberately or accept the double outage. (task t8)
- [unknown_nonblocking] Marlin NVFP4 correctness on `sm_87` is proven only by a single known-answer probe and one tool call. vllm#34694/#49070 report garbled output on this exact fallback; a broader correctness suite before adoption would be prudent and is not yet a task.
- [unknown_nonblocking] The 28 golden .env files and the role test suite were never opened during scope or challenge (scope entry s20). t6's blast radius is asserted, not measured. (task t6)
- [unknown_blocking] MERGE IMPACT (#199, 0.63.0, merged 2026-08-25): the cortex replica pool adds a plural peer family (<PREFIX>`_PEER_ORIGINS` / `_PEER_API_KEYS`), `GATEWAY_SELF_ORIGIN`, a ReplicaCache, live capability fingerprinting and selection — explicitly 'generic across all nine role prefixes'. A TENTH role must therefore also land in lobes/gateway/`_config.py`, `_replicas.py`, `_routing.py` and `_selection.py` plus their new test suites. This CONTRADICTS confirmed honesty condition h35 on boundary c10 ('No gateway, proxy or role-mechanism source file is modified'), which was confirmed against a pre-#199 tree. Either h35's scope is amended to permit the additive tenth-prefix wiring, or t6 cannot be built as specified. (task t6)
- [follow_up] OPPORTUNITY from #199: the Orin's associate and the Spark's worker serve the SAME checkpoint (Nemotron 3.5 Lightning NVFP4). The replica pool's compatibility test is a live-probed fingerprint of served id + quantization + max context + runtime, so the two boxes may be poolable replicas of one lobe rather than two separately-addressed roles. Not evaluated by this frame; it bears directly on c35 (the case for proxying instead) and possibly on v7's tier placement.
