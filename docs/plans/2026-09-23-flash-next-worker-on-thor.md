# Build Plan — flash-next worker on thor

slug: `flash-next-worker-on-thor` · status: `exported` · from frame: `flash-next-worker-on-thor`

> A measured spike boots local-inference-lab/Qwen3.8-Flash-Next-NVFP4 @ada4da32 fully resident (no NVMe PLE mmap) on the Jetson AGX Thor from the digest-pinned official jetson-ai-lab image, benchmarked against a same-day baseline of the incumbent worker, and lands an evidence transcript the operator uses to call GO/NO-GO on re-pointing model=worker

## Tasks

### t1 — Pre-flight: snapshot Thor state, back up ~/.lobes, take a fresh prod postgres dump

- instruction: read-only except the backups and the dump; use ssh thor@thor with env -u `GATEWAY_API_KEY`; discover the postgres superuser from prod-postgres-1's env (`POSTGRES_USER`) and dump via docker exec prod-postgres-1 `pg_dumpall` -U <user> > ~/backups/prod-pg-2026-09-23-pre-flash-next.sql; never print secrets into the transcript
- covers: c19, h7
- acceptance:
  - ~/.lobes/.env and every docker-compose\*.yml have a .bak-2026-09-23-flash-next copy and their sha256 values are recorded in the scratch transcript
  - the scratch transcript records: docker ps with StartedAt for every prod-\* and model-gear-\* container, free -g, swapon --show, dmesg OOM count, df -h /, the Thor gateway's GET /capabilities worker entry, and GET /mesh/roster from the Spark and the Orin
  - a fresh `pg_dumpall` of prod-postgres-1 exists on the Thor outside the container (path + size + timestamp recorded), taken before any lane is stopped
  - git status of lobes-cli records the pre-spike commit, and lobes/roles.py + lobes/catalog.py sha256 are recorded (baseline for h12/h13)

### t2 — Pre-download the pinned checkpoint and pull the image by digest while the incumbent serves

- instruction: run hf download inside a throwaway container with --memory 4g (e.g. python:3.12-slim + uv pip install `huggingface_hub`\[cli\]) mounting ~/.cache/huggingface at /root/.cache/huggingface, env `HF_HUB_DISABLE_XET`=1, --revision ada4da32a583a78aa47299f45a70603c950490b8; pull with docker pull ghcr.io/nvidia-ai-iot/vllm@sha256:512bf772...; do not use the tag
- covers: c34, h21
- acceptance:
  - ~/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/ada4da32a583a78aa47299f45a70603c950490b8/ holds all 35 model-\*.safetensors listed in its model.safetensors.index.json plus config/tokenizer/preprocessor files
  - docker image inspect shows RepoDigests containing ghcr.io/nvidia-ai-iot/vllm@sha256:512bf772c7ef221df1a66ab9c95546d77daeba6ba61723692852e6eb0cae7526
  - the download ran memory-capped with `HF_HUB_DISABLE_XET`=1; incumbent vllm-worker and prod-\* StartedAt are unchanged after it; df -h / after download + pull shows >= 50 GiB free

### t3 — Benchmark the incumbent worker with the shared harness (same-day baseline)

- instruction: use scripts/stream-measure.py copied to the Thor scratch dir, passing the model id explicitly as argv\[2\]; run it from a throwaway container on `lobes_default` against <http://vllm-worker:8000> so both lanes are measured on the identical path; capture the deep prompt file so t7 reuses it
- depends on: t2, t1
- covers: h4, c33, h20
- acceptance:
  - >= 3 decode runs of a short code prompt and >= 3 of a ~9k-token deep prompt against nvidia/Qwen3.6-35B-A3B-NVFP4, each with tok/s and TTFT, stored verbatim with the exact invocation
  - the harness invocation (script, prompts, `max_tokens`, temperature, thinking flag) is saved so t7 can replay it byte-for-byte
  - the Thor gateway log for each run's window is checked for foreign requests; contaminated runs are marked and repeated

### t4 — Prepare the 5-probe script in scratch (text, thinking, tool call, image with negative controls, video)

- instruction: reuse the probe design of docs/evidence/2026-09-10-accept-nvidia-35b-a3b-thor.txt; video: a short generated mp4 (e.g. ffmpeg colour bars then a solid colour) sent as `video_url` base64, PASS if the described content matches; keep everything under the scratchpad, never under lobes/ or tests/
- covers: h10
- acceptance:
  - one script, kept in the session scratchpad (not the repo), emits a literal PASS or FAIL line per probe against a given base URL + model
  - the image probe sends three 64x64 solid PNGs (red, blue, green) and passes only if each colour is named correctly — a wrong-colour answer is a FAIL (negative control)
  - the tool-call probe passes only when the response has a parsed `tool_calls` entry with valid JSON arguments (not prose in content); the thinking probe passes only when reasoning content is separated from content
  - the script is dry-run against the incumbent lane before the cut-over and its output saved (sanity baseline)

### t5 — Cut over: stop every lobes lane except the gateway, point .env at the spike, recreate the gateway

- instruction: run compose from ~/.lobes with the deployment's own -f chain (read it from the gateway container's com.docker.compose.project.`config_files` label); never run compose in the repo's lobes/templates/; gateway recreate: docker compose ... up -d --no-deps --force-recreate gateway
- depends on: t1, t2, t3, t4
- covers: c9, c28
- acceptance:
  - docker compose stop (not down) leaves vllm-worker, vllm-embed, vllm-rerank and any audio lane Exited; the gateway and every prod-\* container keep running
  - .env `WORKER_SERVED_NAME`=local-inference-lab/Qwen3.8-Flash-Next-NVFP4 and `WORKER_MAX_MODEL_LEN`=262144 (only those keys changed, diff against the backup recorded); the gateway is recreated with env -u `GATEWAY_API_KEY` and is healthy
  - the outage start timestamp (incumbent stop) is recorded

### t6 — Boot Flash-Next on the lobes network under the memory abort rule, stepping context down on refusal

- instruction: docker run -d (not -it), --runtime nvidia --gpus all --ipc host, --name flash-next-spike --network `lobes_default` --network-alias vllm-worker; set `HF_HUB_OFFLINE`=1 since t2 pre-downloaded; start the sampler (nohup loop to ~/flash-next-spike/mem.log) BEFORE docker run; the cold load is ~98.6 GiB, allow up to ~20 min before judging a hang
- depends on: t5
- covers: c2, c4, h3, c29, h16, c30, h17, c31, h18, c32, h19
- acceptance:
  - a container named flash-next-spike runs image ghcr.io/nvidia-ai-iot/vllm@sha256:512bf772..., NetworkMode `lobes_default` with alias vllm-worker, no PortBindings, no --rm, no env-file, HF cache mounted at /root/.cache/huggingface with `HF_HOME`=/root/.cache/huggingface, and no `VLLM_PLE_`\* or `VLLM_GDN_DECODE_KERNEL` in its env (docker inspect saved)
  - argv = local-inference-lab/Qwen3.8-Flash-Next-NVFP4 --revision ada4da32a583a78aa47299f45a70603c950490b8 --tokenizer-revision (same sha) --served-model-name local-inference-lab/Qwen3.8-Flash-Next-NVFP4 --gpu-memory-utilization 0.93 --max-num-seqs 1 --max-model-len 262144 --mamba-ssm-cache-dtype bfloat16 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser `qwen3_xml` --hf-overrides {architectures:\[Qwen4ExpForConditionalGeneration\],`model_type`:`qwen4_exp`} --speculative-config {method:mtp,`num_speculative_tokens`:3,model:/opt/qwen38-mtp-model}
  - on a boot refusal the same command is retried at --max-model-len 131072 then 65536 (util fixed at 0.93); each refusal's log excerpt is saved; if 65536 is refused, the spike jumps to t8 and reports
  - a memory sampler (free -m, swap used, dmesg OOM grep, prod-\* StartedAt) logs every 10 s from boot through t7; any abort condition (MemAvailable < 4 GiB sustained 60 s, swap growth > 20 GiB, an OOM kill, a prod-\* restart) stops flash-next-spike immediately and jumps to t8
  - the full docker logs are saved before anything else touches the container, and contain the weights-loaded line, the KV-cache GiB + token-pool lines, and an MTP draft load from /opt/qwen38-mtp-model with no missing-file error
  - if the booted context is below 262144, .env `WORKER_MAX_MODEL_LEN` is set to it and the gateway recreated again

### t7 — Measure and probe Flash-Next through the lanes the mesh actually uses

- instruction: run the harness from the same throwaway container on `lobes_default` as t3; the Spark call uses the Spark gateway key from the Spark's own env, never pasted into the transcript
- depends on: t6
- covers: c8, h2, h5, c20, c22, h15, h20
- acceptance:
  - the t3 harness is replayed byte-for-byte (same prompts, `max_tokens`, temperature, thinking flag) against the candidate: >= 3 short + >= 3 deep runs with tok/s and TTFT; MTP acceptance rate/mean length quoted from the vLLM metrics or logs
  - the t4 probe script runs against model=worker through the Thor gateway AND through the Spark gateway; 5 literal PASS/FAIL lines each
  - iostat -x on the Thor's NVMe during a decode run shows no sustained read throughput (resident PLE evidence, h2)
  - the Thor gateway's GET /capabilities shows worker model local-inference-lab/Qwen3.8-Flash-Next-NVFP4 with the booted context; the Spark's /mesh/roster shows the Thor verified for worker (time to re-verify recorded)
  - first 200 for model=worker via the Spark is timestamped (end of the up-side outage); foreign requests during benchmark windows are checked as in t3

### t8 — Restore the incumbent worker and verify the rollback

- instruction: this task is ALSO the abort path: t6 or t7 jump here on any abort condition or a 65536 refusal — then record which trigger fired; start lanes with compose up -d --no-deps per service, gateway last
- depends on: t7
- covers: c35, h22, c23, h11, c10, h12, c11, h13
- acceptance:
  - flash-next-spike logs are saved (if not already), then the container is removed; ~/.lobes/.env and compose files are restored from the .bak-2026-09-23-flash-next copies and their sha256 equal the t1 values
  - vllm-worker, vllm-embed, vllm-rerank (and audio if it was up in t1) are started and healthy; the gateway is recreated (env -u `GATEWAY_API_KEY`) and /capabilities advertises nvidia/Qwen3.6-35B-A3B-NVFP4 at 262144
  - model=worker answers 200 through the Spark (timestamp = outage end); the Spark and Orin rosters show the Thor verified for worker again
  - every prod-\* container's StartedAt equals its t1 value; git status of lobes-cli shows no change under lobes/ or tests/, and lobes/roles.py + lobes/catalog.py sha256 equal t1's

### t9 — Write the evidence transcript and ship it as a PR

- instruction: follow the style of docs/evidence/2026-09-10-accept-nvidia-35b-a3b-thor.txt; add a pointer line to docs/experiments/qwen3.8-flash-next-nvfp4-thor-reference.md naming this route (c13) only if the operator wants it; version bump via .claude/skills/version-bump/scripts/bump.py patch with the changelog JSON on stdin
- depends on: t8
- covers: c1, h1, c20, h8, c21, h9, c18, h6, c22, h10
- acceptance:
  - docs/evidence/2026-09-23-spike-flash-next-worker-thor.txt exists and contains: image digest, checkpoint revision, full argv, boot-log weights/KV lines, every refused context, incumbent vs candidate benchmark tables, MTP acceptance, 5+5 probe PASS/FAIL lines, memory sampler summary, outage window, restore sha256 pairs + post-restore /capabilities
  - every number in it traces to a t1-t8 capture from this run; nothing quoted from jetson-ai-lab, NemoClaw-Thor or the model card except in a clearly separated 'external claims' note (h1); any missing item says why (h8)
  - the transcript opens with a one-paragraph summary for the operator's GO/NO-GO and states that no GO/NO-GO is made in it; the PR bumps the version (patch), touches nothing under lobes/ or tests/, and links the spec docs/specs/2026-09-23-flash-next-worker-on-thor.md

## Risks

- [unknown_nonblocking] an abort condition fires mid-boot or mid-measure (memory, OOM, prod restart): t8 runs early and the transcript records a partial result instead of the full table (task t6)
- [unknown_nonblocking] mesh members may take up to ~3 heartbeat intervals (3 x 60 s) or longer to re-verify the Thor's changed worker fingerprint, extending the outage past lane health; observed in t7/t8, not controlled (task t7)
- [unknown_nonblocking] the official image or checkpoint may not honour --revision/--tokenizer-revision for the processor/preprocessor files, or the draft config at /opt/qwen38-mtp-model may reference files absent at ada4da32; t6's boot log is the first place this shows (task t6)
- [unknown_nonblocking] thermal/power behaviour under a ~114 GiB resident load at MAXN is unmeasured (frame v9); a thermal throttle would depress the candidate's tok/s — tegrastats is sampled alongside the memory log to make it visible (task t7)
