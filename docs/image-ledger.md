# Image ledger — which container is this, where did it come from, what did it run

Every container image this fleet pins, in one place: **digest, engine and
version, the machine and arch it was validated on, the model(s) it serves, its
build recipe, and the evidence transcript that validated it.**

## Why this file exists

Image provenance was scattered: **10 image pins across
`lobes/templates/fleet/docker-compose.yml`**, 18 more digest mentions across
`docs/*.md`, and 5 `Dockerfile.*` templates, with no index tying them together.
That drift is not hypothetical — it produced a real error the day this file was
written. `docs/lfm2.5-1.2b-hand.md` describes the shared vLLM pin as
`0.23.1rc1.dev672`; that is the **superseded** digest. The shipped pin has been
`0.26.1rc1.dev942+g5a4c8d992` since the qwen3.8-cortex-upgrade, and a spec was
drafted against the wrong number before `docs/vllm-nightly-migration.md:424`
caught it.

**A per-model doc's version note describes the digest that doc was written
against.** This ledger describes the digest the fleet runs.

## Rules

1. **Digests, never tags.** A floating tag is recorded as a floating tag and
   flagged, not silently treated as a pin.
2. **Arch is not a family.** "Blackwell" is not one target: the Spark is
   `sm_121`, the Thor is `sm_110`, the Orin is `sm_87`. An image that installs
   is not an image that has kernels — see the `cu128`/`sm_110` case (#145).
3. **A row with no evidence link is UNVALIDATED**, and says so. Per #108,
   nothing here may claim validated without a transcript under `docs/evidence/`.
4. **Failed recipes keep their row.** An image that did not work, recorded with
   why, is the most useful row in the file for whoever tries next.
5. **Version is read from inside the running container**, never inferred from a
   tag or a digest.

---

## Active pins

### `vllm/vllm-openai@sha256:8bd082c2…` — the shared fleet nightly

| | |
|---|---|
| digest | `sha256:8bd082c274fae025b7079498fe1da65182ba1d4c2188c0f5a68c1042c38c3695` |
| engine | vLLM `0.26.1rc1.dev942+g5a4c8d992` |
| arch | arm64; exercised on `sm_121` (Spark) and `sm_110` (Thor) |
| serves | `cortex` (Qwen3.8-27B-NVFP4), `embedder`, `reranker`, `embed-deep`, `hand` (LFM2.5-1.2B), `worker` (Nemotron Lightning), `associate` (Nemotron Lightning) |
| recipe | upstream pull, no local build |
| knob | `VLLM_NIGHTLY_IMAGE` (per-lane overrides: `HAND_IMAGE`, `WORKER_IMAGE`, `ASSOCIATE_IMAGE`) |
| dated | 2026-08-19 |
| evidence | `docs/evidence/2026-08-19-spike-qwen3.8-official-nightly-spark.txt`; sm_110 in `docs/evidence/2026-08-20-spike-nightly-sm110-thor.txt` |

**One digest, six services, three boxes.** Changing it is a fleet event, not a
lane change. It reaches any box that re-renders from this template.

### `ghcr.io/nvidia-ai-iot/llama_cpp@sha256:f7c67c10…` — the llama.cpp lane

| | |
|---|---|
| digest | `sha256:f7c67c102b08252e963f9e5f92c3a36554c8f69305eb7ea257c6cd12e24c3191` |
| engine | llama.cpp build `38406d597` (10373), CUDA 13, arm64, `CUDAARCHS=87;110` |
| arch | `sm_87` (Orin) exercised; `sm_110` compiled in, **not** exercised |
| serves | `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M` (opt-in `llamacpp` compose profile) |
| knob | `LLAMACPP_IMAGE` |
| dated | 2026-08-23 |
| evidence | `docs/evidence/2026-08-23-spike-qwen38-gguf-llamacpp-orin.txt` |
| status | **configured, NOT validated** — the lane has no acceptance transcript |

Measured on it: 2.61 tok/s decode at `MODE_30W`, **8.46 tok/s / 253.84 tok/s
prefill at MAXN**. The power mode is part of the row because omitting it
produced a whole superseded-numbers section in
`docs/qwen3.8-27b-gguf-llamacpp.md`.

### `lobes/vllm-gemma4:local` — the Gemma 4 lane (locally built)

| | |
|---|---|
| tag | `lobes/vllm-gemma4:local` — **local build, not a registry pin** |
| recipe | `lobes/templates/fleet/Dockerfile.vllm-gemma4` |
| base | `vllm/vllm-openai@sha256:7c5a10e9…` (vLLM `0.23.1rc1.dev672`) |
| serves | `senses` (Gemma 4 12B), `muse` (Gemma 4 31B), `multimodal-coder` |
| knobs | `MULTIMODAL_IMAGE`, `MUSE_IMAGE`, `MULTIMODAL_CODER_IMAGE` |
| evidence | `docs/evidence/2026-07-17-accept-muse-tool-calling-thor.txt` (31B tool calling) |

⚠ **This lane is still on the superseded base.** It did not move with the fleet
nightly, so two vLLM versions run side by side today.

### `nvcr.io/nvidia/vllm:26.04-py3` — NGC ARM64/Blackwell

| | |
|---|---|
| pin | **FLOATING TAG — not a digest.** The only unpinned image in the fleet. |
| serves | `vllm-minor` (opt-in 4B), `vllm-middle` (opt-in 14B), and the legacy single-model `lobes/templates/docker-compose.yml` |
| status | UNVALIDATED as a pin; the tag can move under the fleet without notice |

### Audio overlay and support images

| image | recipe | serves | note |
|---|---|---|---|
| `scitrera/dgx-spark-vllm:0.16.0-t4` | `Dockerfile.parakeet` | `stt` (Parakeet) | **No `sm_87` kernels** — crash-loops on the Orin (measured 2026-07-17). Recorded as a base-image fact, not a budget question. |
| `nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04` | `Dockerfile.chatterbox` | `tts` (Chatterbox) | CUDA 13 chosen because cu128 ships no `sm_110` SASS and no PTX (#145). |
| `python:3.12-slim` | `Dockerfile.gateway`, `Dockerfile.realtime` | gateway, realtime bridge | CPU only, no arch concern. |

---

## Superseded

| digest | engine | why it moved | still referenced by |
|---|---|---|---|
| `sha256:7c5a10e9a8b3…` | vLLM `0.23.1rc1.dev672` | replaced by `8bd082c2…` in the qwen3.8-cortex-upgrade (t5), which flipped the primary from Qwen3.6-27B-NVFP4 to Qwen3.8-27B-NVFP4 in the same change | **still the live base** of `Dockerfile.vllm-gemma4`; also cited as current by `docs/lfm2.5-1.2b-hand.md` and `docs/qwen3-reranker-0.6b.md` — both stale |

## Considered, not adopted

| image | why | source |
|---|---|---|
| `ghcr.io/ggml-org/llama.cpp@sha256:f74f5805…` | upstream llama.cpp server-cuda; the NVIDIA Jetson build was chosen instead for its `CUDAARCHS=87;110` coverage | comment at `docker-compose.yml:283` |

---

## Deferred — recipes specified but not built

Both rows come from `docs/specs/2026-08-27-flash-next-on-vllm.md`, whose
execution was **deferred 2026-08-27, not abandoned**. They are recorded here so
resuming needs no re-derivation.

### Stage 1 — the fleet nightly move

| | |
|---|---|
| target | vLLM main/nightly, arm64, `sm_110` |
| blocker | **vLLM 0.29.0 does not exist.** Latest release is `v0.28.0` (2026-08-26); `recipes.vllm.ai`'s "0.29.0+" names main. There is no version to pin — only a digest. |
| unknown | `cu129` `sm_110` SASS coverage is **unverified**. `nightly-aarch64` and `cu129-nightly-aarch64` exist and are arm64; that is arch availability, not kernel coverage. Gate with `cuobjdump --list-elf` before booting. |
| method | apply as a **`.env` override**, never by editing the template default — override rolls back in one line; a template edit is a repo revert *and* pushes an untested image at every re-rendering box |
| must re-test | the Thor's four validated `sm_110` divergences (cortex `kv_cache_dtype=auto` #109; embed/rerank `TRITON_ATTN` #105; rerank `enforce_eager`) — validated against `0.23`, not automatically true on a newer nightly |
| watch | replica-pool `max_model_len`. `_replicas.py`'s `runtime` field is engine-grained (`vllm`/`llamacpp`), so a version change cannot unpool a pair — but a window trim on the Thor while the Spark holds 262144 **silently** stops pooling, with no error. |
| baseline | benchmark the incumbent cortex **before** the swap — 12.1 tok/s on this Thor (`docs/evidence/2026-08-20-accept-cortex-local-thor.txt`). Per `docs/model-switch-playbook.md` that number is unrecoverable afterwards. |

### Stage 2 — Qwen3.8-Flash-Next GGUF via `vllm-gguf-plugin`

| | |
|---|---|
| target | stage-1 image + `vllm-gguf-plugin` built from source (`uv pip install -e . --no-build-isolation`, CUDA toolkit required), pinned by digest, plugin recorded by **commit sha** not branch |
| model | `unsloth/Qwen3.8-Flash-Next-GGUF` — 125B MoE + 51B n-gram, 6B active |
| unknown | the plugin's tested quants are `Q6_K / Q8_0 / IQ4_XS / Q4_K_M / Q4_0` — **neither `UD-IQ1_M` nor `UD-Q3_K_XL` is on the list**; its architecture list stops at Qwen 2.5/3 with no `qwen4exp`; multi-shard GGUF is undocumented; and the `repo:quant` addressing form is unverified for vLLM |
| footprint | **file size ≠ footprint on vLLM.** llama.cpp mmaps; vLLM loads and dequantizes, plus runtime buffers and CUDA graphs. Against 122 GiB the *load-time peak* decides the outcome, and `UD-IQ1_M` (74.5 GB) may be the only rung that loads. |
| no-op here | `VLLM_PLE_CPU_OFFLOAD=1` frees nothing on unified memory — host RAM and GPU RAM are one pool on Thor and Spark |
| abort | a wall-clock timeout per boot attempt, written down first. Precedent: `docs/evidence/2026-08-20-spike-lightning-thor-no-go.txt` — an indefinite `sm_110` hang on a hybrid state-space decode path, and Flash-Next is 3-of-4 layers Gated DeltaNet. |
| bar | ≥ 25 tok/s decode at MAXN **and** prefill/TTFT in vLLM's territory, not llama.cpp's (~64 tok/s prefill, 610 s TTFT at 32768 — `docs/evidence/2026-08-26-accept-orin-associate.txt`) |

**Why not native:** nothing fits. BF16 335.28 GiB, official `Qwen/…-FP8`
172.78 GiB, `RadixArk/…-NVFP4` 135 GB (routed experts only, published for
SGLang) — all above this box's 122 GiB. The ~35 GB PLE table is the floor.
GGUF is currently the only format that fits.

---

## Keeping this current

This file is only worth its bytes if it does not drift. The spec proposes a test
that greps `lobes/templates/fleet/docker-compose.yml` for `sha256:` and fails on
any digest with no row here — the same enforcement the goldens use. Until that
lands, every image change should update this file in the same PR.
