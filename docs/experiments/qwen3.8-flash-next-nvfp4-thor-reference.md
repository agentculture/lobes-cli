# Qwen3.8-Flash-Next NVFP4 on a Thor — the NemoClaw-Thor reference recipe

**Status: REFERENCE ONLY. Not served, not reproduced here.** This fleet has not
built these images or booted this checkpoint. Every number below is **the
recipe author's own measurement on their own Jetson AGX Thor**, quoted from
[pastoriomarco/NemoClaw-Thor](https://github.com/pastoriomarco/NemoClaw-Thor)
at commit `9c9c0873` (2026-09-06). Per #108 none of it may be cited as
validated on a fleet box until a transcript lands under `docs/evidence/`.

**Decision (2026-09-13): record, don't run.** The Thor keeps serving `worker`
through the `thor-worker` shape. Running this recipe would mean stopping every
lobes lane there, which takes `model=worker` offline for the whole mesh, since
the Spark forwards `worker` to the Thor. The recipe is kept here so that
resuming it later needs no re-derivation.

Sibling doc:
[`qwen3.8-flash-next-gguf-llamacpp-vllm.md`](qwen3.8-flash-next-gguf-llamacpp-vllm.md),
the 2026-08-27 evaluation that ruled this checkpoint out. This doc **corrects
that doc's footprint premise** — see
[Why the 2026-08-27 "doesn't fit" was wrong](#why-the-2026-08-27-doesnt-fit-was-wrong).

## Sources

| source | what it carries |
|---|---|
| `serving/docs/QWEN38-FLASH-NEXT-FAST-THOR.md` | the recommended recipe, measurements, caveats |
| `serving/docs/QWEN38-FLASH-NEXT-NVFP4-THOR.md` | the earlier Triton-GDN baseline the fast recipe supersedes |
| `serving/start-qwen38-flash-next-fast.sh` | the exact `docker run` / `vllm serve` invocation |
| `serving/docker/Dockerfile.qwen38-flash-next-sm110` | overlay 1: the SM110 MoE-backend gate |
| `serving/docker/Dockerfile.qwen38-flash-next-gdn-sm110` | overlay 2: the fused GDN decode kernel and PLE I/O |
| `serving/docker/patches/qwen38-flash-next-sm110-moe.py`, `patches/thor-gdn/` | the patches both overlays apply |
| [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX) | the upstream DGX Spark preview vLLM build the recipe starts from |

All paths are in NemoClaw-Thor unless a link says otherwise.

## The checkpoint

| fact | value | source |
|---|---|---|
| repo / revision | `RadixArk/Qwen3.8-Flash-Next-NVFP4` @ `7b719225242aacd3dbd3f9407468c2ee9a9d2594` | recipe pin; also HF HEAD as of 2026-09-13 |
| size | **126.0 GiB** over 419 files; the FP8 PLE table ships as `model-plefp8-*.safetensors` shards of 4.84 GiB | HF API, read 2026-09-13 |
| architecture | `Qwen4ExpForConditionalGeneration` — 48 layers, 512 experts, `mtp_num_hidden_layers: 1` | `config.json` at that revision |
| modality | `vision_config` present, pipeline `image-text-to-text` | `config.json`, HF API |
| context | `max_position_embeddings: 262144` | `config.json` |
| quantization | native NVFP4 routed experts, BF16 protected layers, FP8 PLE table | recipe |
| drafter | none needed — MTP is embedded in the checkpoint | recipe |
| access | ungated | HF API |

The recipe never tested vision. The checkpoint declares it; whether the preview
image serves it on sm_110 is unknown.

## How it fits and runs: six mechanisms

### 1. The PLE table is memory-mapped from NVMe, not held in memory

The checkpoint's n-gram PLE table is the part no expert-only quantization can
shrink. The preview vLLM maps it from disk (`VLLM_PLE_MMAP=1`, 14 reader
workers, no prewarm) and gathers only the rows a step actually needs. This is
why a 126 GiB checkpoint serves on a 122 GiB unified-memory board at
`gpu_memory_utilization=0.90`. The recipe accordingly requires **fast local
NVMe** for the model cache: the table is read during inference, not just at
load.

### 2. PLE reads are tuned for random access

Overlay 2's `patch_ple_io.py` maps the table with `MADV_RANDOM`, which stops the
kernel's useless sequential read-ahead. `VLLM_PLE_MMAP_FAST_ROWS=0` routes decode
through the existing parallel-read path as well. The author's read-only
microprobe — 100 random 64-row gathers and one 32768-row gather, page cache
cleared between variants:

| mapping / gather | mean 64-row lookup | 32768-row lookup |
|---|---:|---:|
| original / serial small batches | 36.22 ms | 3561 ms |
| `MADV_RANDOM` / serial | 10.17 ms | 422 ms |
| `MADV_RANDOM` / parallel | 2.06 ms | 414 ms |

All variants returned byte-identical rows (same SHA256). This is an I/O probe,
not a tok/s result — and it depends on the author's NVMe, whose model the recipe
does not state.

### 3. A fused GDN decode kernel compiled for SM110a

Three of every four layers are Gated DeltaNet. The preview image's fused CUDA
GDN MTP decode kernel ships **no SM110 code object**, so the first recipe fell
back to Triton (`VLLM_GDN_DECODE_KERNEL=triton`). This fleet hit the same gap
on the 27B cortex; `lobes/profiles/builtin/thor.toml` carries the same
`triton` conjunct.

Overlay 2 closes the gap. It takes four files from vLLM at `082cf021`, the
source associated with the merged
[vLLM PR #53835](https://github.com/vllm-project/vllm/pull/53835) ("Build fused
GDN MTP decode for SM110", merged 2026-09-05), and builds them with
`TORCH_CUDA_ARCH_LIST=11.0a`. The build registers them in a separate
`thor_gdn` op namespace and redirects only the Python dispatch. It then sets
`VLLM_GDN_DECODE_KERNEL=cuda`. The upstream numerical tests
(`test_fused_gdn_post_conv.py`, 22 of them, including recurrent-state rollback
and ragged batches) passed on the author's Thor. Those tests use numerical
tolerances; they are not a quality eval. GDN **prefill** and QSA stay on their
Triton implementations.

### 4. The FlashInfer CUTLASS MoE backend is unblocked on SM110

The preview vLLM excludes compute capability 11.x from
`flashinfer_cutlass_moe.py` (citing flashinfer#3134), and its automatic backend
picked an SM100 vLLM CUTLASS kernel that failed on the Thor. Overlay 1's patch
adds `is_device_capability_family(110)` to that gate. It refuses to run unless
the expected gate text appears exactly once. The launcher then selects the
backend explicitly with `--moe-backend flashinfer_cutlass`. The patch's
rationale: bundled FlashInfer 0.6.17 already maps SM110 to its SM100 CUTLASS
path.

### 5. Embedded MTP at n=3

`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` drafts from
the checkpoint's own head. MTP=3 is the default because it measured faster **per
request**. MTP=2 measured higher **aggregate throughput** (see below). The QSA
backend still rebuilds metadata between draft steps, so the preview does not
support fully fused multi-step drafting.

### 6. Serialized compile cache bypassed

One restart failed with CUDA Xid 13 (illegal instruction) during sampler warmup
after loading an AOT artifact. It was not an OOM. The launcher now sets
`VLLM_DISABLE_COMPILE_CACHE=1` and keeps `VLLM_USE_AOT_COMPILE=1`. That bypasses
the serialized cache without going eager or losing CUDA graphs, at the cost of
extra compile work on every boot. The author calls it a verified workaround, not
a root cause, and has not soak-tested restarts.

## The image: three local layers, every revision pinned

None of these is a published image. Each is built locally, in order:

| tag | built from | pin |
|---|---|---|
| `nemoclaw-thor/qwen38-flash-next-vllm:sm110` | `blazux/qwen3.8-Flash-DGX`, `--build-arg DET_ARCH=110a` | commit `4b723de2e2c465d866738b57ae64bde6e8c07744`; its Dockerfile pins its base by digest |
| `…:sm110-flashinfer-moe` | `Dockerfile.qwen38-flash-next-sm110` (mechanism 4) | NemoClaw-Thor `9c9c0873` |
| `…:sm110-gdn-cuda` | `Dockerfile.qwen38-flash-next-gdn-sm110` (mechanisms 2, 3) | vLLM `082cf021b7ef96e4819e386846ea34e5ef21c655` |

The author's final running image ID:
`sha256:c4800004609fffc26f9c5f6ed90671bee76ae918261e3b93824672d85303b462`.

**Upstream has moved past the pin.** As of 2026-09-13, `blazux/qwen3.8-Flash-DGX`
main is 32 commits ahead of `4b723de2`, including a block-fp8 MTP shim its own
docs call temporary (pending vllm#55513). A reproduction should use the recipe's
pins. Moving to a newer upstream commit is a separate, measured change.

If this fleet ever builds these images, `docs/image-ledger.md` is where the
shas and the resulting image ID go — `lobes/vllm-gemma4:local` is the precedent
for an unpublished local build.

## The serve invocation

From `serving/start-qwen38-flash-next-fast.sh`, defaults shown:

```bash
docker run -it --pull never --name qwen38-flash-next-fast \
  --runtime nvidia --gpus all --ipc host --network host --shm-size 16g \
  -v "$HF_CACHE:/hf" -v "$VLLM_CACHE:/root/.cache/vllm" \
  -v "$TORCH_CACHE:/root/.cache/torch" -v "$FLASHINFER_CACHE:/root/.cache/flashinfer" \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  -e VLLM_DISABLE_COMPILE_CACHE=1 -e VLLM_USE_AOT_COMPILE=1 \
  -e CUTE_DSL_ARCH=sm_110a -e TORCH_CUDA_ARCH_LIST=11.0a \
  -e VLLM_PLE_MMAP=1 -e VLLM_PLE_MMAP_WORKERS=14 -e VLLM_PLE_MMAP_PREWARM=0 \
  -e VLLM_PLE_MMAP_MADV_RANDOM=1 -e VLLM_PLE_MMAP_FAST_ROWS=0 \
  -e VLLM_QSA_DET_TOPK=1 -e VLLM_QSA_DET_LIB=/opt/llm/kernel-det/_C_det.so \
  -e VLLM_QSA_EXACT_TOPK=0 -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_GDN_DECODE_KERNEL=cuda \
  --entrypoint vllm nemoclaw-thor/qwen38-flash-next-vllm:sm110-gdn-cuda \
  serve "/hf/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594" \
  --served-model-name qwen3.8-flash-next --host 0.0.0.0 --port 8050 \
  --load-format safetensors --max-model-len 262144 \
  --max-num-seqs 4 --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 8192 \
  -cc.cudagraph_mode=PIECEWISE "-cc.splitting_ops=$split_ops" \
  --no-enable-flashinfer-autotune --moe-backend flashinfer_cutlass \
  --kv-cache-dtype auto --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

`$split_ops` is the launcher's JSON list of piecewise-CUDA-graph split points.
Beyond the usual attention/mamba ops, it names three Flash-Next-specific ops:
`vllm::qwen3_8_flash_next_ple_short_conv`, `vllm::qwen3_8_flash_next_qsa_with_output`
and `vllm::ple_mmap_lookup`. `VLLM_QSA_DET_TOPK=1` selects deterministic QSA
top-k, and the recipe also keeps its prefix-cache fixes. The recipe's own
prerequisites: JetPack 7.1, and stopping every other GPU model server first.

## The author's measurements

Three fixed coding prompts, one request at a time, temperature 0, thinking off,
512-token output cap, warmup excluded:

| configuration | per-request decode tok/s | mean |
|---|---|---:|
| Triton GDN, MTP=2, util 0.85, original PLE I/O | 23.96 / 22.99 / 27.96 | 24.97 |
| CUDA GDN, MTP=2, 0.85, original PLE I/O | 25.10 / 24.80 / 28.52 | 26.14 |
| CUDA GDN, MTP=2, 0.90, optimized PLE I/O | 33.84 / 31.31 / 33.81 | 32.99 |
| CUDA GDN, MTP=3, 0.90, optimized PLE I/O | 35.92 / 32.57 / 37.82 | 35.43 |
| **final: MTP=3, compile cache bypassed** | 35.77 / 32.20 / 36.96 | **34.98** |

The fused kernel alone was worth ~4.7%; most of the gain came from the PLE I/O
work plus util 0.90. The probe (`serving/benchmarks/flash-next-gdn-ab.py`)
excludes TTFT and approximates multi-token speculative SSE chunks.

| other figure | value |
|---|---|
| KV at the final boot (0.90, MTP=3, BF16 KV) | 19.58 GiB = **677,323 tokens**; KV-pool concurrency ceiling 2.58x at 262144 |
| KV at an MTP=2 boot | 711,119 tokens; KV-pool ceiling 2.71x |
| Triton baseline KV (0.85, MTP=2) | 12.45 GiB = 440,286 tokens; KV-pool ceiling 1.68x |
| 4 concurrent short prompts, 256 tokens each, MTP=3 | **68.10 tok/s** aggregate; 22.52 tok/s mean per request |
| same, MTP=2 | **73.44 tok/s** aggregate (+7.8%); 22.37 tok/s per request |
| 4 concurrent unequal prompts (68 / 2968 / 11595 / 27068 input tokens), incl. cold prefill + JIT | 28.41 tok/s aggregate, no failure |
| host memory available after the 4-way runs | ~10 GiB (short), ~7.2 GiB (unequal); swap unchanged |
| earlier MTP=2 draft acceptance (Triton baseline, live windows) | 49.52% weighted, mean length 2.03 |
| smoke | a parsed `read_file` tool call; a 25,640-token retrieval returned the right answer twice (13.28 s cold, 1.38 s prefix-cached) |

The `x` figures are **KV-pool concurrency ceilings**: KV tokens divided by
`max_model_len` (262144). They are arithmetic from the boot log, not measured
throughput, and must not be multiplied by a single-stream tok/s.

**Not tested by the author:** any full 256K request, four resident full-length
contexts, a sustained or multi-user load, a quality eval, or vision.

**For scale against this fleet's own measured numbers:** on the same board model,
the 27B cortex measured 12.1 tok/s at 1M
(`docs/evidence/2026-08-20-accept-cortex-local-thor.txt`), and the `worker` lane
(`nvidia/Qwen3.6-35B-A3B-NVFP4`) measured 196.6 tok/s
(`docs/nvidia-qwen3.6-35b-a3b-nvfp4.md`). The author's ~35 tok/s would clear
the ≥25 tok/s bar the 2026-08-27 spec set — but it is a different box, a
different workload, and not our measurement.

## Why the 2026-08-27 "doesn't fit" was wrong

The sibling doc lists `RadixArk/Qwen3.8-Flash-Next-NVFP4` as "135 GB — **no,
misses by ~13 GB**", published for SGLang. It also ruled `VLLM_PLE_CPU_OFFLOAD`
a no-op on a unified-memory board, since host RAM and GPU RAM are one pool.
Both observations are correct as far as they go. The footprint conclusion
drawn from them is not:

- the footprint assumed the whole PLE table must be **resident** — mmapping it
  from NVMe (mechanism 1) makes it disk-backed and paged in on demand, which is
  a different axis from CPU offload;
- a preview vLLM (`blazux/qwen3.8-Flash-DGX`) serves this checkpoint; the
  checkpoint is not SGLang-only in practice;
- the pinned revision measures 126.0 GiB on HF, not 135 GB.

So the GGUF-through-`vllm-gguf-plugin` route that spec chose on paper was not
the only route that fits. That spec's other open items — the vLLM version and
sm_110 kernel coverage — are exactly what this recipe's two overlays answer,
using a preview build instead of a release.

## What running it on a fleet box would take

This was scoped read-only on 2026-09-13 and not pursued:

- **The Thor stops being the `worker` host** for the window. The recipe needs
  0.90 util with no other model servers running, and the captured deployment
  `deployments/jetson-agx-thor__thor-worker/` is the rollback. This is the
  reason for the decision above.
- **Disk**: the Thor's single 1 TB NVMe had 146 GB free, against 126 GiB of
  weights plus three ~30 GB image layers. It would need a cleanup first.
- **Memory headroom** is the author's ~7–10 GiB available after load, before
  counting anything else resident on the box.
- **As a lobes lane**, not a hand launcher, it would touch:
  - `lobes/catalog.py` — a candidate entry; `SupportedModel` has no image field.
  - The fleet compose template — `vllm-primary` has no per-lane image override,
    and no lane passes through the `VLLM_PLE_*` / `VLLM_QSA_*` / compile-cache
    env or `-cc.splitting_ops`.
  - A solo Thor shape.
- It would be a **catalog change, not a new role**, per
  `docs/colleague-stack.md`.

## Open unknowns carried by the recipe itself

- **Mixed-length concurrent prefill OOM.** An upstream vLLM issue allocates a
  dense `num_prefills x max_chunk_length` PLE short-conv buffer; this has OOMed
  at util 0.85 even with low KV use. The 4-way unequal smoke passed, which the
  author says is not a guarantee.
- **Restart stability** after the Xid 13 workaround (mechanism 6).
- **FP8 KV** for four resident full contexts is proposed in the baseline doc and
  unvalidated. There is no FP4 KV in this stack.
- **Storage sensitivity.** Decode speed rides PLE read latency, which depends on
  the NVMe.
- **Vision** is declared by the checkpoint and untested.
