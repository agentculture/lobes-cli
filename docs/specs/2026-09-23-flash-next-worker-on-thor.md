# flash-next worker on thor

> A measured spike boots local-inference-lab/Qwen3.8-Flash-Next-NVFP4 @ada4da32 fully resident (no NVMe PLE mmap) on the Jetson AGX Thor from the digest-pinned official jetson-ai-lab image, benchmarked against a same-day baseline of the incumbent worker, and lands an evidence transcript the operator uses to call GO/NO-GO on re-pointing model=worker
> instruction: the transcript lands at docs/evidence/2026-09-XX-spike-flash-next-worker-thor.txt and carries: image digest, checkpoint revision, full serve argv, boot-log weights/KV-pool lines, incumbent + candidate benchmark tables, probe results, rollback confirmation

## Audience

- the lobes operator deciding whether Flash-Next replaces the Thor's worker, and the mesh peers (the Spark's gateway, Qwen Code / colleague callers of model=worker) whose seat it would take

## Before → After

- Before: the Thor serves worker as nvidia/Qwen3.6-35B-A3B-NVFP4 (196.6 tok/s code probe, 262144, util 0.45, DFlash k=12); Flash-Next exists here only as a reference doc for the 126 GiB RadixArk checkpoint that needs NVMe PLE mmap and a 3-layer local image build, never booted on any fleet box
- After: an evidence transcript records whether the 98.6 GiB checkpoint boots resident at util 0.93 and at which max-model-len (262144, else 131072, else 65536), its KV pool, decode tok/s / TTFT / MTP acceptance beside the incumbent's same-day numbers, and pass/fail on text, thinking, tool-call, image and video probes via model=worker; afterwards the Thor is restored to the incumbent thor-worker lane

## Why it matters

- a 125B/6B-active multimodal worker is a much larger doer than the 35B-A3B; the published image + resident NVFP4 PLE removes both blockers the 2026-09-13 decision cited (local build, NVMe paging), so the remaining question is purely measured cost (speed, memory headroom) — which only a boot answers

## Requirements

- the Flash-Next checkpoint fits RESIDENT in the Thor's unified memory — no `VLLM_PLE_MMAP` / NVMe paging of the n-gram table (operator instruction, this session)
  - instruction: verify in the boot log: no `VLLM_PLE_MMAP`\* env in the container (docker inspect), and the PLE quant method logged is the NVFP4 embedding path; during decode, iostat on nvme0n1 shows no sustained read traffic
  - honesty: the container env carries no `VLLM_PLE_MMAP`\* variable and decode shows no sustained NVMe reads — the table is resident, not paged
- the served image is the published ghcr.io/nvidia-ai-iot/vllm:qwen3.8-next-jetson-thor pulled by digest sha256:512bf772c7ef221df1a66ab9c95546d77daeba6ba61723692852e6eb0cae7526 (stock vLLM nightly 385dce36 + /opt/qwen38-thor.patch + baked MTP draft at /opt/qwen38-mtp-model), launched by hand for the spike and recorded in the transcript (docs/image-ledger.md only on GO) — no local 3-layer NemoClaw build
  - instruction: pull by digest: docker pull ghcr.io/nvidia-ai-iot/vllm@sha256:512bf772c7ef221df1a66ab9c95546d77daeba6ba61723692852e6eb0cae7526; record RepoDigests in the transcript
  - honesty: the image that ran is the pinned digest: docker inspect of the running container's .Image resolves to the manifest sha256:512bf772c7ef221df1a66ab9c95546d77daeba6ba61723692852e6eb0cae7526, not a moved tag
- the spike follows docs/model-switch-playbook.md: benchmark the incumbent worker on today's engine first with the same harness used later for the candidate, pre-download the pinned checkpoint and image while the incumbent still serves, back up ~/.lobes (.env + compose files) before stopping anything, then probe text/thinking/tools/image/video and write the evidence transcript
  - instruction: use scripts/stream-measure.py (or lobes benchmark) against each lane directly; store the invocation verbatim in the transcript; backups at ~/.lobes/\*.bak-2026-09-XX-flash-next
  - honesty: incumbent and candidate were benchmarked by the same script with the same prompts, `max_tokens`, temperature and thinking setting, on the same day on the same box
- the mesh sees the spike: model=worker is offline for EVERY mesh member (Spark, Orin, gateway-only; `LOBES_MESH_SEEDS` names Spark and Orin) from the incumbent lane's stop until a worker lane is healthy again; any lane that comes back under a different served id or `max_model_len` changes the Thor's fingerprint (lobes/gateway/`_replicas.py` `DISQUALIFYING_FIELDS`: `served_id`, quantization, `max_model_len`, runtime), which each member must re-verify before routing
  - honesty: the transcript records the outage window (incumbent stop time -> first 200 for model=worker via the Spark) and the Spark's roster/verification state after the spike
- the Thor gateway reads `WORKER_SERVED_NAME` / `WORKER_MAX_MODEL_LEN` / `WORKER_QUANTIZATION` / `WORKER_SPECULATIVE_CONFIG` / parsers from its PROCESS env (docker inspect model-gear-gateway), so the spike recreates the gateway after the .env edit and again after the restore — with env -u `GATEWAY_API_KEY` on the ssh session (Thor ~/.bashrc exports it) — and `WORKER_MAX_MODEL_LEN` in .env tracks whatever context actually booted (c24), else /capabilities advertises a stale window
  - honesty: after each recreate, GET /capabilities on the Thor gateway shows worker model = the lane actually running and context = its booted max-model-len
- the spike has a memory abort rule protecting the co-resident prod stack: prod-postgres-1 runs with no memory limit and `oom_score_adj` 0 on a box with overcommit 0 and a 59.6 GiB swapfile; util 0.93 leaves ~8.6 GiB of 122.8 GiB for OS + prod + gateway + page cache; if MemAvailable stays below 4 GiB, swap use grows past 20 GiB, dmesg logs an OOM kill, or any prod-\* container restarts, the spike container is stopped at once and the incumbent restored
  - honesty: the transcript carries free -g / swap / dmesg OOM lines sampled through boot and benchmarks, and prod-\* container StartedAt timestamps unchanged before vs after
- the spike container runs WITHOUT --rm (the page's command uses --rm, which would destroy the boot log that is the KV-pool evidence); its full docker logs are saved to the transcript's working dir before removal
  - honesty: the saved log contains the weights-loaded line and the KV-cache/pool lines quoted in the transcript
- the spike container mounts /home/thor/.cache/huggingface at exactly /root/.cache/huggingface with `HF_HOME`=/root/.cache/huggingface (the image's /opt/qwen38-mtp-model symlinks hard-code that path + snapshots/ada4da32a583a78aa47299f45a70603c950490b8), passes the FULL 40-char revision sha, and does NOT load ~/.lobes/.env as an env-file — the lobes .env injects `VLLM_GDN_DECODE_KERNEL`=triton (a cortex `sm_110` workaround) into every lane
  - honesty: docker inspect of the spike container shows no `VLLM_GDN_DECODE_KERNEL` and no `VLLM_PLE_`\* env, and the boot log shows the MTP draft loading from /opt/qwen38-mtp-model without a missing-file error
- benchmarks call scripts/stream-measure.py with the model id passed explicitly (argv\[2\]; its default is hard-coded to nvidia/Qwen3.6-35B-A3B-NVFP4) and the transcript notes any foreign request the gateway logged during each benchmark window — worker runs `max_num_seqs`=1, so one mesh caller (Spark proxy, Qwen Code) queues ahead and skews TTFT
  - honesty: the gateway log for each benchmark window shows only the benchmark's own requests, or the affected runs are marked and repeated
- the 98.6 GiB pre-download runs memory-capped (docker --memory or systemd-run MemoryMax) with `HF_HUB_DISABLE_XET`=1 while the incumbent serves, pinned to the full ada4da32 sha; disk after download + image is checked to stay >= 50 GiB free (196 GiB free today)
  - honesty: incumbent worker uptime and prod-\* uptime are unbroken across the download, and df -h / after it is recorded
- restore is verified, not assumed: after the rollback ~/.lobes/.env and every compose file are byte-identical (sha256) to the pre-spike backups, docker ps shows the incumbent vllm-worker + embed + rerank (+ audio if it was up) healthy, and the Thor gateway's /capabilities advertises nvidia/Qwen3.6-35B-A3B-NVFP4 at 262144
  - honesty: the transcript's last section lists the sha256 pairs and the post-restore /capabilities worker entry

## Honesty conditions

- every number in the transcript was measured on the physical Thor in this run — none quoted from jetson-ai-lab, NemoClaw-Thor, or the checkpoint card (#108)
- the transcript quotes the boot log's own 'Available KV cache memory' / KV-pool token lines and the host's free -g at steady state — the 15 GiB figure is replaced by the measured one
- lobes/roles.py `ROLE_RESPONSIBILITIES`/`ROLE_FORBIDDEN` for worker are byte-identical before and after
- no Orin/associate lane is touched and lobes/catalog.py is unchanged by the spike
- the operator named the GO/NO-GO as theirs (c15); the Spark proxies model=worker to the Thor per the 2026-09-11 evidence
- matches the live Thor .env read 2026-09-23 (`WORKER_MODEL`=nvidia/Qwen3.6-35B-A3B-NVFP4, util 0.45, 262144) and docs/experiments/qwen3.8-flash-next-nvfp4-thor-reference.md's reference-only status
- the transcript contains each listed item or states explicitly why it is missing (e.g. a boot refusal) — no silently dropped probe
- both 2026-09-13 blockers are gone for THIS checkpoint: the image is published (ghcr digest) and the PLE table is resident NVFP4 (config `ple_embedding_dtype`=nvfp4 + qwen38-thor.patch's packed GPU PLE method)
- the 5 probes each ran against the candidate lane and each has a literal PASS/FAIL line; the image probe used a negative control (e.g. solid-colour PNGs naming the wrong colour must fail)
- git status in lobes-cli shows no change under lobes/ or tests/ from the spike, and docker ps on the Thor shows the prod-\* containers' uptime unbroken across the spike window
- docker inspect of the spike container shows NetworkMode=`lobes_default` and empty PortBindings; ss -ltn on the Thor shows no listener on :8000 from it

## Success signals

- the spike ends with a transcript under docs/evidence/ containing >= 3 measured decode runs for BOTH incumbent and candidate, the boot-log KV-pool token count, and a pass/fail line for each of 5 probes (text known-answer, thinking, tool call, image vs negative control, video) — and model=worker answers 200 through the Spark after the spike ends

## Scope / boundaries

- the worker role contract in lobes/roles.py does not change: Flash-Next is multimodal (image+video) and a coder, matching #244's widened contract (forbidden only `final_decision`/`security_decision`); image/video evidence status drops to DECLARED/UNMEASURED (#108) until probed on this lane
- associate is untouched: Lightning carries its own `role_hint`=associate, so re-pointing worker's catalog default does not drag the Orin's associate lane (the #244 regression lobes/roles.py:148-200 documents)
- the spike changes no file under lobes/ or tests/ and no packaged template; on the Thor it touches only the lobes lanes (stopped/restarted) and ~/.lobes backups — the prod-\* culture-nodes/postgres/minio containers keep running throughout
- the spike container publishes no host port and does not use --network host (the page's command does): it joins only the `lobes_default` network with alias vllm-worker, so every request reaches it through the Thor gateway's auth gate, never unauthenticated over the tailnet

## Non-goals

- no lobes-lane code in this spec: the `WORKER_HF_OVERRIDES` / `WORKER_MAMBA_SSM_CACHE_DTYPE` compose slots (worker lane lacks both; primary has --hf-overrides at docker-compose.yml:245, associate has --mamba-ssm-cache-dtype at :1745) are deferred to a post-GO follow-up spec
- no new shape in this spec: a solo worker shape for the Thor (0.93 cannot co-reside with thor-worker's embedder+reranker+audio) is deferred to the post-GO follow-up; thor-worker.toml stays the deployed + rollback shape
- no new Colleague role, no catalog entry and no deletion in this spec; if GO, the follow-up adds Flash-Next as a `role_hint`=worker catalog entry and demotes nvidia/Qwen3.6-35B-A3B-NVFP4 to a candidate (cite-don't-delete)
- the NemoClaw-Thor RadixArk recipe (3-layer local build, `VLLM_PLE_MMAP`, fused GDN overlay) is not pursued here; docs/experiments/qwen3.8-flash-next-nvfp4-thor-reference.md stays a reference and gains, at most, a pointer to this route

## Assumptions

- the checkpoint is local-inference-lab/Qwen3.8-Flash-Next-NVFP4 (98.6 GiB, 34-36 shards) — NOT RadixArk/Qwen3.8-Flash-Next-NVFP4 (126.0 GiB) that docs/experiments/qwen3.8-flash-next-nvfp4-thor-reference.md covers; the ~27 GiB saving is its NVFP4 PLE table (config `ple_embedding_dtype`=nvfp4) vs RadixArk's FP8 PLE shards, which is what makes a resident fit plausible
- memory at util 0.93 is the recipe's claim, not a measurement here: 0.93 x 122 GiB ~ 113.5 GiB budget vs ~98.5 GiB of tensors (export `tensor_bytes` 105,798,973,864) leaves ~15 GiB for KV, MTP draft, activations and CUDA graphs; every other lobes lane on the Thor is stopped to fund it (c27) while the non-lobes prod stack (~0.66 GiB resident) stays up; the boot log's KV pool is authoritative, and a refusal is handled by stepping context down at fixed util (c24)

## Scope exploration

- `s1` — `operator instruction (this session)`: Thor worker may be dropped and re-pointed to Qwen3.8 Flash-Next per jetson-ai-lab.com/models/qwen3-8-flash-next; fit directly in memory, not SSD mmap
  - seeds: `c2`
- `s2` — `HF API local-inference-lab/Qwen3.8-Flash-Next-NVFP4 vs RadixArk/... (read 2026-09-23)`: local-inference-lab HEAD 7c4f1bc1 = 98.62 GiB/50 files; RadixArk @7b719225 = 125.96 GiB/419 files; `text_config`.`ple_embedding_dtype`=nvfp4; `quant_method` modelopt `MIXED_PRECISION` (NVFP4 routed experts W4A4 static, MXFP8 attention/shared experts, NVFP4 PLE weight-only); `vision_config` + `video_token_id` present; 262144 native
  - seeds: `c3`
- `s3` — `jetson-ai-lab.com/models/qwen3-8-flash-next page (serve command JSON, thor_t5000::vLLM)`: only module supported by the vLLM engine is `thor_t5000`; command: ghcr.io/nvidia-ai-iot/vllm:qwen3.8-next-jetson-thor local-inference-lab/Qwen3.8-Flash-Next-NVFP4 --gpu-memory-utilization 0.93 --max-num-seqs 1 --mamba-ssm-cache-dtype bfloat16 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser `qwen3_xml` --hf-overrides {architectures:\[Qwen4ExpForConditionalGeneration\],`model_type`:`qwen4_exp`} --speculative-config {method:mtp,`num_speculative_tokens`:3,model:/opt/qwen38-mtp-model}; no --max-model-len (defaults to 262144); no PLE env
  - seeds: `c4`
- `s4` — `ghcr.io registry: nvidia-ai-iot/vllm:qwen3.8-next-jetson-thor manifest + config blob + patch layer`: digest sha256:512bf772..., arm64, 40 layers / 9.01 GiB compressed, created 2026-09-10; base vllm/vllm-openai:nightly-385dce36; ENV `VLLM_DISABLED_KERNELS`=FlashInferCutlassMxfp8LinearKernel; ENTRYPOINT \[vllm serve\]; qwen38-thor.patch (156 lines) adds Qwen4ExpPLENVFP4EmbeddingMethod ('Packed GPU PLE table') in vllm/models/`qwen4_exp`/nvidia/`ple_layer.py` and an `sm_110` capability gate in `qsa_indexer.py`; -latest tag has the same digest
  - seeds: `c4`
- `s5` — `lobes/templates/fleet/docker-compose.yml vllm-worker (image line ~1370, command ~1529-1550) + lobes/profiles/render.py:123-129`: `WORKER_IMAGE` override exists; command slots are a closed set (MODEL, `SERVED_NAME`, QUANTIZATION, `MAX_MODEL_LEN`, `GPU_MEM_UTIL`, `KV_CACHE_DTYPE`, `ATTENTION_BACKEND`, `MOE_BACKEND`, `LOAD_FORMAT`, `MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`, `CHUNKED_PREFILL`, `ASYNC_SCHEDULING`, `PREFIX_CACHING`, `SPECULATIVE_CONFIG`, `TOOL_CALL_PARSER`, `REASONING_PARSER`); no hf-overrides or mamba-ssm-cache-dtype slot on worker (primary has --hf-overrides at :245, associate has --mamba-ssm-cache-dtype at :1745); render.py maps `hf_overrides`->`HF_OVERRIDES`; template's default `WORKER_MODEL` is still stale Lightning
  - seeds: `c5`
- `s6` — `Thor: docker inspect model-gear-vllm-worker (read-only, 2026-09-23)`: deployed worker entrypoint is \[bash /usr/local/bin/mg-logwrap\] with Cmd 'vllm serve ...', so the official image's own ENTRYPOINT \[vllm serve\] is overridden and does not double up
  - seeds: `c5`
- `s7` — `lobes/profiles/builtin_shapes/thor-worker.toml + lobes/profiles/builtin/thor.toml`: thor-worker hosts \[worker, hand, embedder, reranker, stt, tts\]; worker override util 0.45 / 262144 / fp8 KV / `max_num_seqs` 1 / DFlash k=12 (z-lab/Qwen3.6-35B-A3B-DFlash); thor.toml: hand feasible=false (`sm_110` defect), embedder/reranker 0.06 each; goldens tests/goldens/shapes/thor-`worker__`{thor,spark}.env pin `WORKER_MODEL`
  - seeds: `c6`
- `s8` — `Thor host (ssh thor@thor: free -g, docker stats, df -h, docker ps; read-only 2026-09-23)`: 122 GiB total, 30 GiB available with worker at 13.2 GiB rss; prod-\* culture-nodes/postgres/minio resident ~0.66 GiB; / has 196 GiB free (enough for 98.6 GiB weights + 9 GiB compressed image), 105.9 GB reclaimable images; no Flash-Next snapshot cached, no nemoclaw image; .env `MODEL_GEAR_VERSION`=0.81.1, `WORKER_MODEL`=nvidia/Qwen3.6-35B-A3B-NVFP4, `COMPOSE_PROFILES`=worker, `LOBES_MESH_NAME`=thor; R38.2.2, driver 580, CUDA 13.0
  - seeds: `c7`
- `s9` — `docs/model-switch-playbook.md + docs/nvidia-qwen3.6-35b-a3b-nvfp4.md + deployments/jetson-agx-thor__thor-worker/`: playbook §1/§7: incumbent baseline is unrecoverable after the swap, so benchmark first; §3: update the `SERVED_NAME` in .env or the gateway rewrites to a dead id; §9: rollback recipe before the swap; the thor-worker lock (captured 2026-09-11) is the pre-designated rollback artifact; lobes benchmark / lobes measure / scripts/stream-measure.py exist to re-baseline
  - seeds: `c8`
- `s10` — `lobes/gateway/_mesh_wire.py:65-69 + _replicas.py:196,298-410 + _mesh_routing.py:206-265`: Fingerprint carries `served_id`/`served_name` among `DISQUALIFYING_FIELDS`; pooling requires identical fingerprints; a changed id is re-probed, not silently kept
  - seeds: `c9`
- `s11` — `lobes/roles.py:404-420,520 (ROLE_RESPONSIBILITIES/ROLE_FORBIDDEN worker)`: worker responsibilities include `repo_action`, `image_understanding`, `video_understanding`; forbidden = (`final_decision`, `security_decision`); widened by #244 because the checkpoint gained a ViT
  - seeds: `c10`
- `s12` — `lobes/roles.py:148-200 (associate/worker served-name + role_hint history)`: issue #244 t1 once dragged associate's default along with worker's; the catalog now gives Lightning `role_hint`=associate so a worker re-point is isolated
  - seeds: `c11`
- `s13` — `lobes/catalog.py:70-130,1029-1114 (SupportedModel + current worker entry)`: SupportedModel has `hf_overrides`, `moe_backend`, `speculative_config`, `hf_revision` fields but no image or extra-env field; current worker entry `role_hint`=worker, modelopt, MTP n2 default (shape overrides with DFlash); docs/colleague-stack.md: a checkpoint swap is a catalog change, a new role is effectively irreversible
  - seeds: `c12`
- `s14` — `docs/experiments/qwen3.8-flash-next-nvfp4-thor-reference.md (2026-09-13, reference-only)`: RadixArk 126 GiB fits only by NVMe-mmapping the FP8 PLE table at util 0.90; author's 34.98 tok/s MTP=3 unreproduced; decision then: record, don't run, because the Thor must stop serving worker
  - seeds: `c13`
- `s15` — `HF API local-inference-lab/Qwen3.8-Flash-Next-NVFP4 commits + revision ada4da32 vs 7c4f1bc1 + image history RUN snapshot_download`: image build resolved revision=`model_info`(model).sha at 2026-09-10 = ada4da32 (PTQ, 34 shards, arch `Qwen3_8FlashNextForConditionalGeneration`) and symlinked /opt/qwen38-mtp-model/<shard> -> hub cache snapshots/<that rev>/<shard>; 2026-09-16 commit b13380df 'Publish ... quantization-aware distilled checkpoint' replaced it (36 shards, MTP tensors in model-00034-of-00036); export-manifest.validation.`full_model_serving` = 'not run'
  - seeds: `q1` (question, resolved)
- `s16` — `operator instruction (this session, mid-/think)`: 'we can drop other lanes for it' — the Thor's embedder/reranker/audio lanes may be stopped to fund the Flash-Next worker; the prod-\* non-lobes stack is not a lobes lane and stays
  - seeds: `c27`
- `s17` — `challenge pass / adjacent-systems lens: Thor model-gear-gateway container env + lobes/gateway/_config.py:920, server.py:1087, _replicas.py:196-201`: gateway carries `WORKER_`\* in process env; mesh `DISQUALIFYING_FIELDS` = `served_id`, quantization, `max_model_len`, runtime — the spike changes `served_id` and possibly `max_model_len`, so the advert and fingerprint go stale unless the gateway is recreated
  - seeds: `c28`
- `s18` — `challenge pass / failure-mode + data-loss lens: Thor docker inspect prod-postgres-1, /proc/sys/vm/overcommit_memory, swapon --show`: postgres:17-alpine prod data, HostConfig.Memory=0, OomScoreAdj=0, restart unless-stopped; `overcommit_memory`=0, swappiness 60, 59.6 GiB file swap with 7.1 GiB used; prod-backup-1 running
  - seeds: `c29`
- `s19` — `challenge pass / observability lens: jetson-ai-lab serve command (sudo docker run -it --rm --pull always ...)`: --rm + --pull always: logs vanish on exit and the tag can move; the spike pins the digest (c4) and keeps the container until logs are saved
  - seeds: `c30`
- `s20` — `challenge pass / security lens: Thor docker inspect model-gear-vllm-worker networks (lobes_default aliases [model-gear-vllm-worker vllm-worker])`: the incumbent lane has no host port, reachable only via the gateway on `lobes_default`; the page's --network host would expose an unauthenticated vLLM on every Thor interface
  - seeds: `c31`
- `s21` — `challenge pass / hidden-dependency lens: Thor docker inspect model-gear-vllm-worker env + image history RUN snapshot_download`: every lobes lane gets the whole .env as env (worker shows `VLLM_GDN_DECODE_KERNEL`=triton, `VLLM_MODEL`=unsloth/Qwen3.8-27B-NVFP4 etc.); the image's draft links resolve only under /root/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/<full sha>/
  - seeds: `c32`
- `s22` — `challenge pass / concurrency lens: scripts/stream-measure.py:10 + thor .env WORKER_MAX_NUM_SEQS=1`: the harness defaults MODEL to the incumbent id; with one sequence slot, any live mesh caller during a run contaminates it
  - seeds: `c33`
- `s23` — `challenge pass / operations lens: memory orin-uncapped-process-oom-kills-associate + Thor df -h`: an uncapped hf-xet download OOM-killed a serving engine on the Orin 2026-09-13; the Thor has swap and 30 GiB available today but runs the incumbent + prod during the download
  - seeds: `c34`
- `s24` — `challenge pass / reversibility lens: Thor ~/.lobes listing (docker-compose.yml.bak-* precedents) + HF cache (incumbent + z-lab DFlash drafter cached)`: the incumbent's weights and DFlash drafter stay cached so restore is a restart, not a download; prior backups use docker-compose.yml.bak-<tag> naming; there is no lobes CLI on the Thor, so restore is compose via ssh
  - seeds: `c35`
- `s25` — `challenge pass / overlooked-actors lens: Thor .env LOBES_MESH_SEEDS (spark:8001, orin:8000)`: the outage and fingerprint change reach the Orin and any gateway-only member too, not only the Spark
  - seeds: `c9`
- `s26` — `challenge pass / security lens (clean): local-inference-lab ada4da32 file list`: no .py files in the checkpoint, so no remote code executes; the page's command omits --trust-remote-code and the spike keeps it omitted — clean; residual: the image itself is third-party-published (nvidia-ai-iot), trusted by digest only
- `s27` — `challenge pass / unexamined: Spark + Orin gateways' live reaction to a Thor fingerprint change; thermal behaviour under a 114 GiB resident load; prod-* peak memory`: not probed in this pass — the Spark/Orin rosters were not queried, no thermal or prod workload profile was read; these surface only during the spike itself

## Decisions

- revision pinned to ada4da32 (PTQ export matching the image's baked MTP draft) via --revision / --tokenizer-revision; QAD HEAD is out of this scope (operator, 2026-09-23)
- no tok/s gate: the spike records decode/TTFT/MTP acceptance and probes as an evidence transcript; GO/NO-GO is a later operator decision (operator, 2026-09-23)
- the first step is a hand-launched spike of the official command on the Thor with the lobes stack stopped, not a lobes-lane code change; `WORKER_`\* slots, the solo shape and the catalog entry land only after a GO (operator, 2026-09-23)
- served-model-name is the raw HF id local-inference-lab/Qwen3.8-Flash-Next-NVFP4, the fleet convention (operator, 2026-09-23)
- on a boot refusal the spike keeps util 0.93 and steps max-model-len 262144 -> 131072 -> 65536, recording every refused value; if 65536 is refused too it stops and reports to the operator (operator, 2026-09-23)
- after measuring, the Thor is restored to the incumbent thor-worker lane (nvidia/Qwen3.6-35B-A3B-NVFP4) pending the GO/NO-GO (operator, 2026-09-23)
- the spike lane runs behind the Thor's gateway: attached to the lobes compose network with network alias vllm-worker, .env `WORKER_SERVED_NAME` temporarily set to the raw id (backed up and restored), so model=worker is probed through the Thor gateway and through the Spark proxy (operator, 2026-09-23)
- every other lobes lane on the Thor (embedder, reranker, audio stt/tts) may be dropped to fund the Flash-Next lane — for the spike window, and as the intended hosting if GO; the gateway stays up (operator, 2026-09-23)
- memory abort rule for the spike: MemAvailable below 4 GiB sustained, swap growth over 20 GiB, any logged OOM kill, or any prod-\* container restart stops the spike container and restores the incumbent; a fresh prod postgres dump is taken before the Flash-Next boot (operator, 2026-09-23)

## Hard questions

- Which revision do we serve? The image's /opt/qwen38-mtp-model symlinks were grafted at build (2026-09-10) against snapshot ada4da32 (PTQ export, 34 shards model-000NN-of-00034); HF HEAD since 2026-09-16 is b13380df/7c4f1bc1, a QAD-distilled re-export with 36 shards (-of-00036) whose manifest says `full_model_serving`: not run. The page's command pulls HEAD unpinned, so the baked MTP draft points at files that never download. Options: (a) --revision ada4da32 (matches the image, PTQ quality); (b) HEAD QAD + re-graft the draft dir (a thin derived image or a bind-mounted draft config/index); (c) HEAD QAD with MTP off. (resolved: operator 2026-09-23: pin PTQ revision ada4da32a583a78aa47299f45a70603c950490b8 (matches the image's baked MTP draft); QAD HEAD is a later, separate change)
- Are the 4 GiB MemAvailable / 20 GiB swap-growth thresholds the right abort lines for this box, and is a fresh prod-backup-1 dump required before the boot? (resolved: operator 2026-09-23: thresholds stand (MemAvailable < 4 GiB sustained, swap growth > 20 GiB, any OOM kill, any prod-\* restart => abort + restore); take a fresh prod postgres dump before the boot)

## Open parks

- [unknown_nonblocking] vision/video on `sm_110` through the official image: declared by config (`vision_config`, `video_token_id`) and by the jetson-ai-lab page, unmeasured here
- [unknown_nonblocking] decode tok/s, TTFT and MTP acceptance for this checkpoint+image on the Thor: no measurement exists for local-inference-lab's checkpoint on any Thor; the page publishes none
- [unknown_nonblocking] `qwen3_xml` vs the fleet's `qwen3_coder` tool parser: the page uses `qwen3_xml`; whether strict tool calling / the `qwen3_coder_thinking` plugin (cortex-only today) applies to this lane is unexplored
- [unknown_nonblocking] whether each mesh member re-verifies the Thor's changed worker fingerprint promptly (announce/heartbeat 60 s x 3) or serves 503 `role_unverified` / 404 for longer — unprobed; observe during the spike
- [unknown_nonblocking] thermal and power behaviour of a ~114 GiB resident 125B MoE under sustained decode on the Thor at MAXN — unmeasured
- [follow_up] the -latest tag equals the pinned digest today; if NVIDIA republishes it with a re-grafted draft for QAD, option (b) of q1 may become a digest bump rather than a derived image
- [follow_up] post-GO lobes-lane integration: `WORKER_HF_OVERRIDES` + `WORKER_MAMBA_SSM_CACHE_DTYPE` slots, a solo Thor worker shape, a catalog entry, goldens (tests/goldens/shapes/thor-`worker__`\*.env), docs (nvidia-qwen3.6-35b-a3b-nvfp4.md, colleague-stack.md, CLAUDE.md, lobes/explain/catalog.py), a deployment-lock re-capture
- [follow_up] post-GO integration hazard: the fleet .env's `VLLM_GDN_DECODE_KERNEL`=triton (thor.toml cortex workaround) would leak into a Flash-Next worker lane via `env_file`; the integration must scope or unset it and measure which GDN decode kernel the official image wants

## Resolved vagueness

- [unknown_blocking] whether util 0.93 boots on THIS Thor with the prod-\* stack resident and ~15 GiB of non-weight headroom — only the boot log's KV pool answers it; fallback util unknown until measured — resolved: the spike itself answers it from the boot log's KV pool; on refusal, step context down 262144 -> 131072 -> 65536 at fixed util 0.93, then stop and report (decision c24)
