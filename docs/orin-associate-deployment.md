# Running Nemotron 3.5 Lightning on a Jetson AGX Orin — the deployment, in full

The deployed `orin-associate` shape now runs at the checkpoint's **native
1,048,576-token (1M) window**. This is the copy-pasteable version of what
`docs/evidence/2026-09-13-measure-associate-budget-orin-1m.txt` (the
NVFP4-vs-W4A16 A/B) and `docs/evidence/2026-09-13-accept-orin-associate-1m.txt`
(the rendered shape, live) measured. Nothing here is machine-specific: every
value is either a literal you can use as-is or a `${VAR}` the scaffold
substitutes. **HISTORY:** the previous 128,000-token deployment
(`docs/evidence/2026-08-26-accept-orin-associate.txt`) is superseded and kept
below for the record, not deleted.

**Measured on a Jetson AGX Orin 64GB (Ampere sm_87, 61.34 GiB unified, zero
swap), 2026-09-13, at 1,048,576 tokens:** KV pool 2,899,067 tokens (2.76x
capacity ceiling at 1M) on the rendered shape; cold 1,040,073-token needle
PASS at TTFT 2390.17 s / decode 7.98 tok/s at depth (A/B); 2-session, 12-turn
agentic run PASS (24/24 tool calls, 6/6 recall) at wall 119.1 s direct /
198.4 s through the gateway.

---

## The short version: plain `docker run`, at the native 1,048,576-token window

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
  --max-model-len 1048576 \
  --gpu-memory-utilization 0.70 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 8192 \
  --mamba-backend flashinfer \
  --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 \
  --trust-remote-code \
  --speculative-config '{"method": "dspark", "model": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark", "num_speculative_tokens": 5, "kv_cache_dtype": "bfloat16"}'
```

Measured live, 2026-09-13 (`docs/evidence/2026-09-13-measure-associate-budget-orin-1m.txt`):
"Available KV cache memory: 17.34 GiB" / "GPU KV cache size: 2,889,456
tokens" / "Maximum concurrency for 1,048,576 tokens per request: 2.76x" (a
KV-pool capacity ceiling, not measured throughput — `--max-num-seqs 2` caps
the lane at two concurrent sequences regardless of what the pool could hold).

### HISTORY (superseded 2026-09-13): the original 128,000-token `docker run`

The pre-1M command used `--max-model-len 128000`,
`--gpu-memory-utilization 0.80` (solo, no other lanes resident), no
`--max-num-seqs`, and `--max-num-batched-tokens 16384` — kept here for the
record, not deleted (cite-don't-delete):

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

96.8 tok/s decode at depth 0 and 121.1 tok/s at depth 512 with DSpark, against
54.8/54.7 plain — and ~21x the Qwen3.8-27B GGUF the board ran before, all at
the (superseded) 128,000-token window.

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
5. **`--gpu-memory-utilization 0.70` at 1M (or `0.80` solo at 128K,
   HISTORICAL) assumes a specific co-resident set.** With a different set of
   lanes resident it will be refused at boot. See the budget table.
6. **A cold request near 1M tokens can take 2,000-2,500 s TTFT/wall.** That
   exceeds the 600 s `GATEWAY_READ_TIMEOUT` every mesh member but the Orin
   keeps by default — see "Cold-request timeout support" below.

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

The `:-16384` and `:-65536`/`:-0.30` fallbacks above are the **template's**
generic defaults, used only when a deployment's `.env` sets no value at all —
they are not the `orin-associate` shape's numbers. The shape's own
`[overrides.associate]` (`lobes/profiles/builtin_shapes/orin-associate.toml`)
sets `max_num_batched_tokens = 8192`, `gpu_mem_util = 0.70`, `max_model_len =
1048576` and `max_num_seqs = 2` for the current, 1M-window deployment — see
the budget table below.

## Budget — pick by what else is on the board

`gpu_memory_utilization` is a fraction of the WHOLE device, so every co-resident
byte comes out of KV. All measured on the same 61.34 GiB board.

**Current (2026-09-13), the native 1,048,576-token window, associate-first
boot order** (`docs/evidence/2026-09-13-measure-associate-budget-orin-1m.txt`,
`docs/evidence/2026-09-13-accept-orin-associate-1m.txt`):

| Co-resident | util | max_model_len | max_num_batched_tokens | Result |
|---|---:|---:|---:|---|
| embedder + reranker (A/B, hand not started) | **0.70** | 1,048,576 | 8,192 | KV 17.34 GiB · pool 2,889,456 · 2.76x |
| embedder + reranker (rendered shape, accepted) | **0.70** | 1,048,576 | 8,192 | KV 17.4 GiB · pool 2,899,067 · 2.76x |

`hand` is **not** part of this co-resident set at the 1M budget — the shape's
`hosts` no longer names it (see "Five things that will bite you" and the
memory-headroom note below).

**HISTORY (superseded 2026-09-13): the 128,000-token window, gears-first boot
order** (`docs/evidence/2026-08-25-measure-associate-budget-orin.txt`,
`docs/evidence/2026-08-26-accept-orin-associate.txt`):

| Co-resident | util | Result |
|---|---:|---|
| nothing (solo) | **0.80** (HISTORICAL — superseded) | KV 20.99 GiB · pool 2,395,428 · 18.71× — **with** DSpark |
| nothing (solo) | 0.80 (HISTORICAL — superseded) | KV 23.35 GiB · pool 3,806,000 · 29.73× — plain |
| embedder + reranker | 0.63 (HISTORICAL — superseded) | KV 10.95 GiB · pool 1,249,280 · 9.76× (HISTORICAL — superseded) |
| + hand | 0.63 (HISTORICAL — superseded) | **REFUSED** — `hand` holds 5.84 GiB, not the 3.68 its util implies |
| + hand | **0.56** (HISTORICAL — superseded) | KV 9.35 GiB · pool 1,524,000 · 11.91× (HISTORICAL — superseded) |
| embedder + reranker | 0.70 (HISTORICAL — superseded) | **REFUSED** by 0.05 GiB (the vendor's value) |

At the 128,000-token, gears-first budget, `max_num_batched_tokens` was
**16,384** (HISTORICAL — superseded); the current 1,048,576-token,
associate-first budget uses **8,192**.

## Throughput

**Current (2026-09-13), needle depths at the native 1,048,576-token window**
(`docs/evidence/2026-09-13-measure-associate-budget-orin-1m.txt`, NVFP4 arm,
cold requests; 256 forced completion tokens):

| Prompt depth | TTFT | decode at depth |
|---:|---:|---:|
| 128,073 | 89.81 s | 29.42 tok/s |
| 250,073 | 214.27 s | 19.71 tok/s |
| 1,040,073 | 2,390.17 s | 7.98 tok/s |

2-session, 12-turn agentic run (~118K max prompt tokens): wall 119.1 s direct
to the lane; 198.4 s through the Orin gateway on the rendered shape (the
cause of that gap is not isolated — candidates are the gateway hop, the
refreshed compose, and prefix-cache state after two cold 1M prefills).

**HISTORY (superseded 2026-09-13): depth sweep at the 128,000-token window**
(`docs/evidence/2026-08-26-accept-orin-associate.txt`), 128 output tokens,
unique prompts (repetitive text hits the prefix cache and inflates TTFT
wildly):

| Depth | DSpark | plain | gain |
|---:|---:|---:|---:|
| 0 | 96.82 tok/s | 54.75 | 1.77× |
| 512 | **121.12** | 54.70 | **2.21×** |
| 2,048 | 90.58 | 54.57 | 1.66× |
| 8,192 | 93.86 | 54.00 | 1.74× |
| 32,768 | 59.05 | 52.13 | 1.13× |

Draft acceptance 35–64%, mean accepted length 2.77–4.18 of 5; the per-position
rate decays (~0.93/0.81/0.67/0.49/0.28), which is why the gain shrinks with
depth. The drafter costs ~37% of the KV pool. (All figures in this HISTORY
block are at the superseded 128,000-token window.)

## Memory headroom at the 1M window — quote both figures, not just one

This is a **zero-swap** board, so host memory headroom is the binding
constraint, not just GPU memory. Two runs measured it, and per **approved
deviation d2** (operator, 2026-09-13, recorded on issue #260) both figures
must be quoted together — never 2,588 MiB alone:

- **2,588 MiB** — the A/B's whole-run minimum available host memory
  (`docs/evidence/2026-09-13-measure-associate-budget-orin-1m.txt`, at
  03:16:25 UTC during the cold 1.04M-token prefill).
- **1,925 MiB** — the SHIPPED shape's whole-run minimum available host
  memory (`docs/evidence/2026-09-13-accept-orin-associate-1m.txt`, at
  07:09:49Z, during a cold 1.03M-token streamed prefill).

Both runs had associate, embed, rerank and the gateway container resident,
so the gap is **not explained by an extra resident container**. The cause is
**not isolated** — do not invent one. Treat the lower figure (1,925 MiB) as
the operating headroom for capacity planning, and the higher figure (2,588
MiB) as the earlier, narrower A/B measurement.

## The memory-capped side-process rule (zero-swap board)

On 2026-09-13, an **uncapped** side process — an `hf-xet` download running
during A/B preparation — triggered a global kernel OOM that killed the
associate engine (recorded on issue #260; two kernel "Killed process"
entries in `dmesg` bracket the incident). This board has **zero swap**, so an
unbounded process can starve every resident container at once, not just
itself. The rule going forward: run any side work (downloads, benchmarking
scripts, anything not part of the served lanes) **memory-capped**
(`docker run --memory <limit> ...` or the local equivalent) and, for any
Hugging Face download specifically, with **`HF_HUB_DISABLE_XET=1`** set —
the xet download path was the trigger here and has no cap of its own.

## The gear first-start race

`model-gear-vllm-embed` fails its **first** start whenever the gears are
started (embed + rerank together) immediately after the associate-class
engine goes healthy, with `ValueError: No available memory for the cache
blocks`. It recovers on Docker's own `restart=unless-stopped` retry — this
is the existing compose lane's restart policy doing its job, not a new
defect. Seen in both the A/B (3 failed starts across the run) and the
accepted rollout (embed `RestartCount` 1 after the run). `rerank` restarted
once in the A/B without that error line, and zero times in the accepted
rollout. Expect one embed restart on a fresh boot at this budget; it is not
a sign of instability.

## Cold-request timeout support is Orin-gateway-only (decision c30)

A cold associate request near the 1,048,576-token ceiling can take
**2,000-2,500 s** of TTFT/wall (measured: 2,390.17 s non-streamed at
1,040,073 tokens, 1,971.42 s to first byte streamed at 1,030,073 tokens —
vLLM v0.27.1 sends no bytes at all before the first generated token, so
streaming does not dodge a read timeout). Every other mesh member keeps the
template's default `GATEWAY_READ_TIMEOUT=600` (600 s), which such a request
would blow through. Only the **Orin card** (`lobes/profiles/builtin/orin.toml`
`[host_env]`) raises this to **`GATEWAY_READ_TIMEOUT=7200`** — an Orin-only
override, decision c30. Consequently, a cold associate request whose TTFT
exceeds a member's `GATEWAY_READ_TIMEOUT` is a **supported path only through
the Orin's own gateway** — routing such a request through a different mesh
member (including via the mesh join's auto-wiring to a proxied `associate`)
is not covered by any measurement here and would time out at that member's
600 s default.

## Verify it

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"associate","messages":[{"role":"user","content":"Capital of France? One word."}],"max_tokens":1024}'
```

Give it **plenty of `max_tokens`**: this is a thinking model, and a small budget
is consumed by the reasoning trace before any content is emitted — you get an
empty `content` and `finish_reason: "length"`, which looks like a broken model
and is not.

For a >= 1M-token cold request, use the **Orin's own gateway** and a client
timeout well above 2,500 s (see "Cold-request timeout support" above), not
`localhost:8000` directly unless you accept the same wait on a bare
connection.
