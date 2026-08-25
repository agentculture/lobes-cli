# Running Nemotron 3.5 Lightning on a Jetson AGX Orin — the deployment, in full

This is the copy-pasteable version of what
`docs/evidence/2026-08-26-accept-orin-associate.txt` measured. Nothing here is
machine-specific: every value is either a literal you can use as-is or a
`${VAR}` the scaffold substitutes.

**Measured on a Jetson AGX Orin 64GB (Ampere sm_87, 61.34 GiB unified, zero
swap):** 96.8 tok/s decode at depth 0 and 121.1 tok/s at depth 512 with DSpark,
against 54.8/54.7 plain — and ~21× the Qwen3.8-27B GGUF the board ran before.

---

## The short version: plain `docker run`

If you just want the model up, without lobes:

```bash
docker run -d --name nemotron-associate \
  --runtime nvidia \
  -p 127.0.0.1:8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/vllm:/root/.cache/vllm" \
  --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  vllm/vllm-openai:v0.27.1 \
  --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --served-model-name associate \
  --quantization modelopt \
  --kv-cache-dtype bfloat16 \
  --max-model-len 128000 \
  --gpu-memory-utilization 0.80 \
  --mamba-backend flashinfer \
  --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --max-num-batched-tokens 16384 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 \
  --trust-remote-code \
  --speculative-config '{"method": "dspark", "model": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark", "num_speculative_tokens": 5, "kv_cache_dtype": "bfloat16"}'
```

### Five things that will bite you

1. **The image must be `v0.27.1` or newer.** Older vLLM builds do not know the
   `dspark` speculative method and **refuse to start** — they do not fall back
   to plain decode. The error lists the methods they do support; `dspark` is
   absent.
2. **The DSpark repo id contains `-NVFP4`.** NVIDIA's own published Jetson
   recipe omits it and names a repo that does not exist; vLLM then fails with
   `Invalid repository ID`.
3. **`--kv-cache-dtype bfloat16`, not fp8.** The checkpoint declares
   `kv_cache_quant_algo: FP8`, but sm_87 has no FP8 KV path.
4. **`-p 127.0.0.1:8000:8000`, not `--network host`.** The vendor recipe uses
   `--network host` with no API key and CORS open. On a tailnet-connected box
   that publishes an unauthenticated 30B endpoint to every peer — which is not
   hypothetical: during our own spike, two peers queried it within seconds.
5. **`--gpu-memory-utilization 0.80` assumes the model is alone on the board.**
   With other lanes resident it will be refused at boot. See the budget table.

---

## The lobes version

```bash
lobes init --shape orin-associate --apply     # renders .env + compose overrides
lobes fleet up --apply
```

Two keys are **operator-typed** in the deployment's `.env` — they cannot be
declared by the shape (a shape has no env mechanism) and must not be declared
by the card (it would leak them onto shapes that drop associate):

```bash
ASSOCIATE_IMAGE=vllm/vllm-openai:v0.27.1
ASSOCIATE_SPECULATIVE_CONFIG="'--speculative-config={\"method\": \"dspark\", \"model\": \"nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark\", \"num_speculative_tokens\": 5, \"kv_cache_dtype\": \"bfloat16\"}'"
GATEWAY_API_KEY=<your inbound bearer token>
```

They are a **matched pair**: arming the speculative config without pinning the
image fails at boot.

## The compose service, verbatim

From `lobes/templates/fleet/docker-compose.yml` — every knob is a `${VAR}` with
the shipped default after `:-`:

```yaml
  vllm-associate:
    image: ${ASSOCIATE_IMAGE:-${VLLM_NIGHTLY_IMAGE:-vllm/vllm-openai@sha256:...}}
    container_name: model-gear-vllm-associate
    profiles: [associate]                 # opt-in: absent unless COMPOSE_PROFILES names it
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - { driver: nvidia, count: all, capabilities: [gpu] }
    ipc: host
    ulimits:
      memlock: { soft: -1, hard: -1 }
      stack:   { soft: 67108864, hard: 67108864 }
    env_file:
      - path: .env
        required: false
    environment:
      - HF_HOME=/root/.cache/huggingface
      - TOKENIZERS_PARALLELISM=false
      - VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
    volumes:
      - ${HF_CACHE:-${HOME:-/root}/.cache/huggingface}:/root/.cache/huggingface
    expose:
      - "8000"                            # NOT `ports:` — only the gateway is published
    command: >-
      vllm serve ${ASSOCIATE_MODEL:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}
      --served-model-name=${ASSOCIATE_SERVED_NAME:-...}
      --host=0.0.0.0 --port=8000
      --quantization=${ASSOCIATE_QUANTIZATION:-modelopt}
      --kv-cache-dtype=${ASSOCIATE_KV_CACHE_DTYPE:-bfloat16}
      --max-model-len=${ASSOCIATE_MAX_MODEL_LEN:-65536}
      --gpu-memory-utilization=${ASSOCIATE_GPU_MEM_UTIL:-0.30}
      --mamba-backend=${ASSOCIATE_MAMBA_BACKEND:-flashinfer}
      --mamba-ssm-cache-dtype=${ASSOCIATE_MAMBA_SSM_CACHE_DTYPE:-float16}
      ${ASSOCIATE_MAMBA_CACHE_STOCHASTIC_ROUNDING:---enable-mamba-cache-stochastic-rounding}
      --mamba-cache-philox-rounds=${ASSOCIATE_MAMBA_CACHE_PHILOX_ROUNDS:-5}
      --mamba-cache-mode=${ASSOCIATE_MAMBA_CACHE_MODE:-align}
      ${ASSOCIATE_PREFIX_CACHING:---enable-prefix-caching}
      --max-num-batched-tokens=${ASSOCIATE_MAX_NUM_BATCHED_TOKENS:-16384}
      ${ASSOCIATE_SPECULATIVE_CONFIG-}
      --enable-auto-tool-choice
      --tool-call-parser=qwen3_coder
      --reasoning-parser=${ASSOCIATE_REASONING_PARSER:-nemotron_v3}
      --trust-remote-code
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 600s
```

`expose:` rather than `ports:` is the whole security posture: the lane is
reachable only on the compose network, and the **gateway** is the single
published surface, behind `GATEWAY_API_KEY`.

## Budget — pick by what else is on the board

`gpu_memory_utilization` is a fraction of the WHOLE device, so every co-resident
byte comes out of KV. All measured on the same 61.34 GiB board:

| Co-resident | util | Result |
|---|---:|---|
| nothing (solo) | **0.80** | KV 20.99 GiB · pool 2,395,428 · 18.71× — **with** DSpark |
| nothing (solo) | 0.80 | KV 23.35 GiB · pool 3,806,000 · 29.73× — plain |
| embedder + reranker | 0.63 | KV 10.95 GiB · pool 1,249,280 · 9.76× |
| + hand | 0.63 | **REFUSED** — `hand` holds 5.84 GiB, not the 3.68 its util implies |
| + hand | **0.56** | KV 9.35 GiB · pool 1,524,000 · 11.91× |
| embedder + reranker | 0.70 | **REFUSED** by 0.05 GiB (the vendor's value) |

## Throughput

Depth sweep, 128 output tokens, unique prompts (repetitive text hits the prefix
cache and inflates TTFT wildly):

| Depth | DSpark | plain | gain |
|---:|---:|---:|---:|
| 0 | 96.82 tok/s | 54.75 | 1.77× |
| 512 | **121.12** | 54.70 | **2.21×** |
| 2,048 | 90.58 | 54.57 | 1.66× |
| 8,192 | 93.86 | 54.00 | 1.74× |
| 32,768 | 59.05 | 52.13 | 1.13× |

Draft acceptance 35–64%, mean accepted length 2.77–4.18 of 5; the per-position
rate decays (~0.93/0.81/0.67/0.49/0.28), which is why the gain shrinks with
depth. The drafter costs ~37% of the KV pool.

## Verify it

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"associate","messages":[{"role":"user","content":"Capital of France? One word."}],"max_tokens":1024}'
```

Give it **plenty of `max_tokens`**: this is a thinking model, and a small budget
is consumed by the reasoning trace before any content is emitted — you get an
empty `content` and `finish_reason: "length"`, which looks like a broken model
and is not.

