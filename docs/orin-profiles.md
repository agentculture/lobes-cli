# Jetson AGX Orin 64GB — live-validated senses / embedder / reranker profile

**Status: operator-validated on physical hardware, 2026-07-16/17** (issue #127
mesh work). This page records what a real Jetson AGX Orin 64GB Developer Kit
(Ampere **sm_87**, 61.3 GiB unified memory, CUDA 13.2 / driver 595.78) actually
serves, with measured numbers — the #108 discipline: a card earns its tuning
data by booting it, and this boot is what earned Orin's. The repo's built-in
`orin-small` *shape* remains DECLARED/UNVALIDATED (this deployment used
`thor-lobe`, not `orin-small` — see "Shape choice" below).

> **`cortex` is no longer categorically infeasible on this board (2026-08-23).**
> Everything below still describes what this deployment measured and served in
> July 2026, and the NVFP4 statement in it is still true as written — the NVFP4
> exports quantize *activations* to FP4 and need Blackwell. What changed is that
> the line turned out to be about the CHECKPOINT FORMAT, not the role: the
> built-in `orin` card profile now declares `cortex` on a weight-only **GGUF**
> gear served by llama.cpp (`unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M`), which decodes
> on Ampere — the same reason the int4 W4A16 `senses` gear runs here. It cannot
> co-reside with `senses` (MEASURED: ~33 GiB + ~27.6 GiB against 61.3 GiB with
> zero swap), so it comes with its own shape, `orin-cortex`, which drops `senses`
> the way `orin-lobe` drops `cortex`. Both are **DECLARED, UNVALIDATED** — no box
> has booted either. That non-co-residency is now DECLARED DATA the tool acts on
> (`[[exclusive_roles]]` in the card profile), not just prose: a shape-less
> `lobes init` on this board is REFUSED, naming `orin-cortex`/`orin-lobe`, rather
> than rendering a deployment that starts both and OOMs. See
> `docs/qwen3.8-27b-gguf-llamacpp.md`, `docs/machine-profiles.md` and
> `docs/deployment-shapes.md`.

## What this box serves

| role | model | context | gpu_mem_util | result |
|---|---|---|---|---|
| `senses` | `coolthor/gemma-4-12B-it-NVFP4A16` | **131072 (full native 128K)** | **0.45** | KV pool **802,644 tokens**, **6.12×** concurrency at 131,072/request |
| `embedder` | `Qwen/Qwen3-Embedding-0.6B` | 8192 | 0.06 | healthy; 1024-dim embeddings |
| `reranker` | `Qwen/Qwen3-Reranker-0.6B` | 8192 | 0.06 | healthy; **rerank ordering probe passes** |
| `cortex` | — | — | — | `feasible=false` (see "Why no cortex"), referred/proxied to the Spark |
| `stt`/`tts` | — | — | — | dropped locally; `/v1/audio/*` forwards to a peer gateway via `AUDIO_URL` |

Probes passed (2026-07-17, via the gateway on `:8000`): senses known-answer
(text, `model=senses` alias), senses **vision** intake (solid-color image →
correct one-word answer), embeddings (1024 dims), rerank ordering (relevant
document top-ranked, score 0.82). The three core-role correctness probes that
`lobes assess` encodes are therefore all exercised on sm_87.

## The operator profile

There is deliberately no built-in `orin` profile in `lobes/profiles/builtin/`
(#108: no invented knobs for an unbooted card). This deployment declares an
**operator profile** at `<deploy-dir>/profiles/orin.toml`, selected with
`lobes init --profile orin`:

```toml
name = "orin"
summary = "Jetson AGX Orin 64GB (Ampere sm_87, 64 GB unified) — operator profile: senses + pooling gears, no cortex"

[roles.cortex]
feasible = false

[roles.senses]
feasible = true
model = "coolthor/gemma-4-12B-it-NVFP4A16"
gpu_mem_util = 0.45
max_model_len = 131072
quantization = "compressed-tensors"
attention_backend = "TRITON_ATTN"

[roles.embedder]
feasible = true
model = "Qwen/Qwen3-Embedding-0.6B"
gpu_mem_util = 0.06
max_model_len = 8192
attention_backend = "TRITON_ATTN"

[roles.reranker]
feasible = true
model = "Qwen/Qwen3-Reranker-0.6B"
gpu_mem_util = 0.06
max_model_len = 8192
attention_backend = "TRITON_ATTN"
enforce_eager = true
```

### Why these knobs

- **`senses` runs on Ampere because the checkpoint is weight-only FP4.**
  `coolthor/gemma-4-12B-it-NVFP4A16` is compressed-tensors **W4A16** — weights
  dequantise through vLLM's Marlin/FP4 path (compute capability ≥ 8.0), so
  sm_87 qualifies. Activations stay 16-bit; no Blackwell FP4 tensor cores
  needed. The full stack — Marlin GEMM, torch.compile/inductor, the MTP
  (`eagle_head`) draft, TRITON_ATTN — ran clean on sm_87 with the same pinned
  vLLM nightly digest the Spark/Thor fleets use (`vllm/vllm-openai@sha256:7c5a…`,
  vLLM 0.23.1rc1.dev672). vLLM's own arm64 release builds list 8.7 as an
  aarch64-only target, which is why the multi-arch image carries the kernels.
- **`gpu_mem_util = 0.45` is MEASURED, not computed** (the same discipline as
  thor-lobe's 0.30). The first boot at util 0.30 (18.4 GiB of 61.3) was refused
  by vLLM: after weights (~7.7 GiB), the MTP draft head, CUDA graphs
  (0.75 GiB) and runtime overhead, only **2.25 GiB** KV remained where
  `max_model_len=131072` needs **3.08 GiB** (vLLM's estimated max at 0.30 was
  76,352). At 0.45 (~27.6 GiB) the same boot reports **18.86 GiB KV =
  802,644 tokens = 6.12×** concurrency at full context. Do not copy Thor's
  0.30 onto a 64 GB board for 128K.
- **`max_model_len = 131072`** — the checkpoint's full native context. It fits;
  no 64K/32K trim was needed on 64 GB once util was raised.
- **`cortex feasible = false` (why no cortex).** The 27B primary
  (`sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`) is **modelopt NVFP4 W4A4** —
  its quantised *activations* need Blackwell FP4 tensor cores; sm_87 is
  Ampere. This is a hard architecture line, not a memory tradeoff.
- **Pooling gears mirror Thor's Jetson divergences as the conservative first
  boot** (`TRITON_ATTN` both, reranker `enforce_eager`). Thor's FLASH_ATTN
  pooling hang is sm_110-specific and was NOT re-tested on sm_87 — these are
  safe-boot choices, not proven-necessary ones. Relaxing them (and measuring)
  is open follow-up work.

## Shape choice: `thor-lobe`, not `orin-small`

The built-in `orin-small` reference shape (minor + pooling + audio, both heavy
lobes dropped) was NOT what this deployment wanted — the assignment here is
**Orin hosts the Gemma senses lobe** (most important role for this box, user
decision on #127). `thor-lobe` hosts exactly `senses + embedder + reranker +
stt + tts` and drops `cortex`, so it renders the right deployment on this box
when composed over the operator profile:

```bash
lobes init --profile orin --shape thor-lobe --apply
```

Note the shape's own senses overrides (`gpu_mem_util=0.30`,
`max_model_len=131072` — Thor-measured values) overlay the profile at render
time; on this box the rendered `.env` needed `MULTIMODAL_GPU_MEM_UTIL=0.45`
(the Orin-measured value above). `orin-small` itself remains unbooted and
therefore still DECLARED/UNVALIDATED.

> **Superseded by `orin-lobe`.** That hand-patch is exactly what the built-in
> `orin-lobe` shape now removes: it hosts `senses + embedder + reranker`,
> drops `cortex`, drops `stt`/`tts` (this board has no sm_87 Parakeet image —
> see the divergences below), and carries the Orin card's own budget in its
> `[overrides.senses]`, so `lobes init --profile orin --shape orin-lobe`
> renders `MULTIMODAL_GPU_MEM_UTIL=0.45` / `MAX_MODEL_LEN=262144` with nothing
> to patch afterwards. It is itself **DECLARED, UNVALIDATED** — no box has
> booted it — and it renders against the built-in `orin` card profile, whose
> two senses values are a MEASURED-PENDING hypothesis pending a live boot. The
> same render also persists this board's Tegra iowait threshold (below).

## Jetson/sm_87 divergences found live (upstreaming candidates)

1. **csv-mode GPU access.** This Orin's NVIDIA container toolkit (1.19.1,
   `mode = "auto"`) resolves to legacy **csv** mode, where the compose
   template's `deploy.resources.reservations.devices` (`--gpus`-style) request
   fails at container create: *"invoking the NVIDIA Container Runtime Hook
   directly … is not supported. Please use the NVIDIA Container Runtime"*.
   Fix applied on-box: every GPU service's `deploy:` stanza in the deployed
   `docker-compose.yml`/`docker-compose.audio.yml` replaced with
   `runtime: nvidia`. That hand edit did not survive a re-init. (Thor presumably
   works because JetPack 7 ships CDI mode.)

   **Now declared in-tree.** The built-in `orin` card profile carries
   `gpu_access = "runtime"`, and `lobes init --apply` generates
   `docker-compose.gpu.yml` + `docker-compose.gpu-audio.yml` — compose overrides
   that `!reset` each GPU service's `deploy:` stanza and set `runtime: nvidia`
   instead. Written on **every** render (and removed when the resolved card does
   not declare it), so there is nothing left to patch by hand and a re-init is
   now the thing that *restores* the fix rather than reverting it. Compose has
   no conditional-block syntax, so a second compose file — the mechanism
   `docker-compose.shape.yml` already proved — is the only form this can take;
   `lobes fleet files` layers both halves into the `-f` chain automatically. See
   [`machine-profiles.md`](machine-profiles.md) `gpu_access`. **UNVALIDATED
   (#108):** `docker compose config` renders it correctly on this board, which
   proves the compose merge, not a container create — that needs a live boot.
2. **The Parakeet STT base image is Spark-specific.** `Dockerfile.parakeet`'s
   base `scitrera/dgx-spark-vllm:0.16.0-t4` ships a torch with no sm_87
   kernels — NeMo dies at model load with `CUDA error: no kernel image is
   available for execution on the device` (observed live: 8 container
   restarts). Working fix (staged on-box before local audio was dropped):
   rebase the Dockerfile onto the fleet's own pinned vLLM nightly digest
   (whose arm64 torch is sm_87-validated by senses itself) and add
   `ENTRYPOINT []` so its CMD survives the base image's entrypoint. The same
   mismatch plausibly explains stt/tts `ready=false` on the sm_110 Thor.
3. **Unified-memory budgeting is tighter than the util sum.** Each vLLM engine
   consumes CPU-side memory beyond `gpu_mem_util × total` (host buffers plus
   the default 4 GiB `swap_space` per engine). With senses 0.45 + two 0.06
   gears the box sat at **54/61 GiB used with zero swap configured** — the
   util sum (0.57 ≈ 35 GiB) undercounts by ~19 GiB. Leave real headroom, or
   audio sidecars/host workloads will OOM-race the fleet.
4. **Phantom iowait sheds the whole box.** `/proc/stat` here reports ~59%
   iowait — 5 of 8 cores at ~97% — with **zero disk I/O** (`vmstat` bi/bo ≈ 0),
   GPU 0%, cores at minimum clock: the Tegra `sugov:0`/`sugov:4`
   cpufreq-governor kthreads flicker through D state and inflate `nr_iowait`.
   The gateway's pressure policy reads that literally and, at the shipped
   `LOBES_IOWAIT_DEGRADED_THRESHOLD=50`, enters busy mode and **429-sheds every
   full-tier request indefinitely** — on this card, all of `senses`. The
   deployed box survived on an *ephemeral* shell-env override of `100` that any
   `docker compose up` reverts. This is now **declared on the built-in card
   profile** (`lobes/profiles/builtin/orin.toml`'s `[host_env]`), so every
   render writes `LOBES_IOWAIT_DEGRADED_THRESHOLD=100` into `.env` and
   `lobes doctor` names the key on a deployment still carrying `50`. The swap
   guard stays at its shipped `75` — swap is measured honestly here. A
   repo-level fix (PSI-based or disk-I/O-corroborated sampling that does not
   trust `nr_iowait` on Tegra) would retire the declaration.

## Mesh wiring (#127) from this box

- `PRIMARY_PEER_ORIGIN=http://spark.tail0be7e0.ts.net:8001` — cortex referral
  (the same operator-typed value the Thor uses). `PRIMARY_PEER_PROXY=true` +
  `PRIMARY_PEER_API_KEY=<the Spark's inbound GATEWAY_API_KEY>` arm the
  follow-the-referral forward. **VALIDATED LIVE 2026-07-17**: a
  `model=cortex` chat request from this box answers from the Spark's 27B
  (`X-Lobes-Proxied-By: http://spark.tail0be7e0.ts.net:8001`, HTTP 200),
  caller's own credential stripped per the #127 contract.
- `AUDIO_URL=http://100.127.105.72:8001` — `/v1/audio/*` is *configured* to
  forward to the Spark's gateway instead of local sidecars, but **audio
  cannot chain gateway→gateway as of 0.45.0** (found live, 2026-07-17): the
  audio readiness probe GETs `<AUDIO_URL>/v1/health/ready` unauthenticated,
  which a peer *gateway* 401s (its GET /v1/* namespace is bearer-gated) and
  would 404 anyway (that endpoint belongs to the realtime *bridge*, and
  gateways only path-route `/v1/audio/*`) — the probe returns False forever
  and this box answers a permanent 503. Working peer-audio today means
  pointing `AUDIO_URL` at the peer's realtime **bridge** directly (publish
  its port, e.g. `8080`, on the peer's tailnet — the bridge is keyless and
  serves `/v1/health/ready` itself). Audio is also **outside the #127
  pairwise-auth contract** (four core roles only): the forward passes the
  caller's own `Authorization` through verbatim, attaches no per-peer key,
  and `stt`/`tts` get no `hosted_by` referral annotation. First-class audio
  referral/proxy knobs are the phase-2 candidate this measures out.

## Reproducing on another Orin

```bash
uv tool install lobes-cli               # or: uv sync in a checkout
mkdir -p ~/.lobes/profiles              # then write profiles/orin.toml (above)
lobes init --profile orin --shape thor-lobe --apply
# csv-mode GPU access (divergence 1) now needs NO hand edit: the built-in `orin`
# profile declares gpu_access = "runtime", so the init above also generates
# docker-compose.gpu{,-audio}.yml. A CDI-mode Orin overrides the built-in with
# its own <deploy-dir>/profiles/orin.toml leaving gpu_access at the default.
lobes up senses --apply                 # boot order: senses first, then gears
lobes up embedder --apply && lobes up reranker --apply
# `lobes fleet files` prints the -f chain this deployment resolves to — use it
# rather than hand-listing overlays:
docker compose $(lobes fleet files) up -d gateway
```

Boot sequentially — the machine-profiles boot-ordering caveat (concurrent
first boots race vLLM's free-memory measurement on unified boards) applies to
this card too.
