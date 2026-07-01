# vLLM-nightly migration — before-state verification + baselines to beat

**Task t1** of the devague plan
[`lobes-unifies-its-generate-lane-on-one-vllm-nightl`](plans/2026-07-01-lobes-unifies-its-generate-lane-on-one-vllm-nightl.md)
(covers spec claims c3, h6, h7). This is a **verification-only** doc: it records
what the fleet templates and catalog actually pin *today*, cited to exact
`file:line`, and the baselines a future nightly migration has to beat. It makes
**no** change to `docker-compose.yml`, `env.example`, `catalog.py`, or any image
pin — that mutation is scoped to later tasks (t2/t3 spikes → t4 flip → t5–t7
Gemma DSpark → t8 trailing gears → t9 shipped-state docs), per the plan's
`Execution waves`.

## 1. Today's image pins (verified in-repo)

The fleet's generate/embed/rerank vLLM services and the Gemma multimodal gear
are **not** on the same engine today — that asymmetry is the whole reason this
migration exists.

### Primary, embed, rerank, minor, middle — all pin the NGC 26.04-py3 image

Grepped directly from the real template
(`lobes/templates/fleet/docker-compose.yml`), not assumed:

| Service | image pin | file:line |
|---|---|---|
| `vllm-primary` (generate `main` tier) | `nvcr.io/nvidia/vllm:26.04-py3` | `lobes/templates/fleet/docker-compose.yml:24` |
| `vllm-embed` (embedding gear) | `nvcr.io/nvidia/vllm:26.04-py3` | `lobes/templates/fleet/docker-compose.yml:111` |
| `vllm-rerank` (reranker gear) | `nvcr.io/nvidia/vllm:26.04-py3` | `lobes/templates/fleet/docker-compose.yml:164` |
| `vllm-minor` (opt-in 4B, `COMPOSE_PROFILES=minor`) | `nvcr.io/nvidia/vllm:26.04-py3` | `lobes/templates/fleet/docker-compose.yml:234` |
| `vllm-middle` (opt-in legacy 14B, `COMPOSE_PROFILES=middle`) | `nvcr.io/nvidia/vllm:26.04-py3` | `lobes/templates/fleet/docker-compose.yml:307` |

That image tag resolves to **vLLM `0.19.0+nv26.04`** — confirmed by every
per-model doc that serves on it, e.g.
`docs/qwen3.6-27b-text-nvfp4-mtp.md:183` ("Image `nvcr.io/nvidia/vllm:26.04-py3`
(vLLM `0.19.0+nv26.04`)") and `docs/gemma-4-12b-nvfp4.md:238` (`| nvcr.io/nvidia/vllm:26.04-py3 | 0.19.0 | ❌ | ❌ |`,
the last two columns being the two blockers this migration exists to fix: no
native `gemma4_unified` class, no MTP-capable loader for that arch on 0.19.0).
The single-model (non-fleet) `lobes/templates/docker-compose.yml:26` scaffold
pins the same `nvcr.io/nvidia/vllm:26.04-py3` image.

### The Gemma 4 12B `vllm-multimodal` gear already runs vLLM nightly

`vllm-multimodal` does **not** pin `nvcr.io/nvidia/vllm:26.04-py3` — it builds a
custom image instead
(`lobes/templates/fleet/docker-compose.yml:396-399`: `build: {context: ., dockerfile:
Dockerfile.vllm-gemma4}`, `image: ${MULTIMODAL_IMAGE:-lobes/vllm-gemma4:local}`),
and that Dockerfile bases off the **official vLLM nightly image, pinned by
digest**:

```text
lobes/templates/fleet/Dockerfile.vllm-gemma4:24
FROM vllm/vllm-openai@sha256:7c5a10e9a8b3c8642f4d0463a41215176c0dd834b4f0967287c7e3e517cf1be9
```

Per the comment directly above that line
(`lobes/templates/fleet/Dockerfile.vllm-gemma4:18-23`), the digest "resolves to
vLLM 0.23.1rc1.dev672 + transformers 5.12.1 (registers gemma4_unified) on
Blackwell-capable torch, and was live-validated on the DGX Spark GB10 (sm_121)
on 2026-07-01 (#71)". The reason it needs nightly at all: Gemma 4's
`gemma4_unified` arch is early-fusion multimodal with heterogeneous per-layer
head sizes, and the native `Gemma4UnifiedForConditionalGeneration` class that
handles that "only exists in vLLM NIGHTLY (>= 0.23.1rc1)"
(`lobes/templates/fleet/Dockerfile.vllm-gemma4:9`); released vLLM <= 0.22.1
(including the NGC 26.06 base the Dockerfile used before) falls back to the
generic Transformers backend and crashes the full-attention layers' `o_proj`.

### Summary

| Gear | Image today | Engine |
|---|---|---|
| `main` (27B primary) | `nvcr.io/nvidia/vllm:26.04-py3` | vLLM 0.19.0+nv26.04 |
| embed | `nvcr.io/nvidia/vllm:26.04-py3` | vLLM 0.19.0+nv26.04 |
| rerank | `nvcr.io/nvidia/vllm:26.04-py3` | vLLM 0.19.0+nv26.04 |
| minor (opt-in 4B) | `nvcr.io/nvidia/vllm:26.04-py3` | vLLM 0.19.0+nv26.04 |
| middle (opt-in legacy 14B) | `nvcr.io/nvidia/vllm:26.04-py3` | vLLM 0.19.0+nv26.04 |
| `multimodal` (Gemma 4 12B) | `vllm/vllm-openai@sha256:7c5a...` via `Dockerfile.vllm-gemma4` | vLLM 0.23.1rc1.dev672 (nightly) |

Two engines, two vLLM minor versions, on one gateway. Unifying the generate
lane onto one nightly (this plan's t2–t9) removes that split and makes a
same-engine head-to-head (t6) meaningful — today a 27B-vs-12B comparison would
be confounded by engine version, not just model size.

## 2. Baselines to beat

These are the numbers a nightly-served 27B primary and a nightly-served Gemma
12B (with MTP/DSpark wired, t5) have to match or beat for the migration to be
worth it. Recorded here from the source docs, not re-measured — t2/t6 do the
re-measurement.

### 27B primary (`sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`), today's 0.19.0 engine

Source: `docs/qwen3.6-27b-text-nvfp4-mtp.md`.

- **Decode throughput ≈ 19 tok/s** — the promotion benchmark measured
  **19.1 tok/s** (batch=1, greedy, 512 tok forced,
  `docs/qwen3.6-27b-text-nvfp4-mtp.md:193`), and the 128K-context re-run
  measured **18.3 tok/s** (1000 tok forced, `docs/qwen3.6-27b-text-nvfp4-mtp.md:227`);
  the production-flags `lobes benchmark` run under the compose template landed
  at **18.7 tok/s** (`docs/qwen3.6-27b-text-nvfp4-mtp.md:293`). Summary line:
  "**18.7–19.1 tok/s decode (~2.4× the archived baseline 27B's ~8 tok/s)**"
  (`docs/qwen3.6-27b-text-nvfp4-mtp.md:13-14`).
- **MTP draft acceptance: 72–79 %.** 72.2 % at promotion
  (`docs/qwen3.6-27b-text-nvfp4-mtp.md:195`), 73.3 % at 128K context
  (`docs/qwen3.6-27b-text-nvfp4-mtp.md:229`), 74.0 % at longer context
  (`docs/qwen3.6-27b-text-nvfp4-mtp.md:257`), and **78.6 %** with
  `--enable-auto-tool-choice` on under the production compose
  (`docs/qwen3.6-27b-text-nvfp4-mtp.md:291`) — the summary line states
  "**72–79 % MTP acceptance**" (`docs/qwen3.6-27b-text-nvfp4-mtp.md:307`).
- Measured on `nvcr.io/nvidia/vllm:26.04-py3` (vLLM `0.19.0+nv26.04`),
  `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'`
  (`docs/qwen3.6-27b-text-nvfp4-mtp.md:183`).

### Gemma 4 12B multimodal gear, no spec-decode

Source: `docs/gemma-4-12b-nvfp4.md`.

- **Decode throughput ≈ 23 tok/s, no draft.** "the benchmark below (~23 tok/s
  single-stream, no draft) *is* that no-spec baseline"
  (`docs/gemma-4-12b-nvfp4.md:151-152`); the measured table reads **"~23 tok/s
  (batch=1, greedy — 21.7–23.3 across balanced/prompt-heavy; 23.0 tok/s
  sustained over 1,500 forced tokens in 65.2 s)"** (`docs/gemma-4-12b-nvfp4.md:286`),
  with `Speculative decoding: none` on the same line block
  (`docs/gemma-4-12b-nvfp4.md:292`).
- This baseline already out-decodes the 27B primary single-stream (~23 vs
  ~18–19 tok/s, `docs/gemma-4-12b-nvfp4.md:294-297`) despite having no
  spec-decode boost at all — the gear is smaller (12B vs 27B). The DSpark
  speculative-decoding wiring this plan adds (t5/t6) has to clear this ~23
  tok/s bar to be worth shipping default-on (t7's verdict gate).
- Measured on `lobes/vllm-gemma4:nightly-audio` (vLLM `0.23.1rc1.dev672`,
  native `Gemma4UnifiedForConditionalGeneration`, `TRITON_ATTN`,
  `compressed-tensors` NVFP4, `--max-model-len 8192`) — i.e. the Gemma baseline
  is *already* a nightly-engine number, unlike the 27B baseline above.

## 3. The generate lane carries real mesh traffic (not hypothetical)

This benchmark isn't academic — the `main` tier is the model the deployed
`lobes` mesh agent itself thinks with, right now, in production:

- `culture.yaml:8` declares the lobes agent's runtime backend as
  `model: vllm-local/sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` — every response
  the `lobes` agent posts to the Culture mesh is generated by the exact 27B
  primary this doc benchmarks. `docs/gateway-fleet.md:400-401` confirms the
  resolution path: "`culture.yaml`'s `model: vllm-local/sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`
  resolves through the gateway on `:8000` as the default."
- The gateway's tier-alias contract is what makes that traffic identifiable
  per-gear: "Callers send `model=main|minor|multimodal` (or the back-compat
  aliases `hard|cheap|normal`) instead of a full model id"
  (`docs/gateway-fleet.md:95-96`) — `model=main` addresses the 27B primary
  benchmarked in §2, `model=multimodal` addresses the Gemma 12B gear
  benchmarked in §2.
- `docs/gemma-4-12b-nvfp4.md:157-158` names the `multimodal` lane's audience
  explicitly: "lobes operators/maintainers and **the Culture mesh that
  consumes** the `multimodal` (Gemma 4 12B) generate lane — i.e. anyone calling
  `model=multimodal` or the back-compat `model=normal`."

So both gears this plan puts on one engine are load-bearing for real mesh
callers today, not idle candidates — a regression in either (lost MTP,
gibberish, a dropped tool-call parser) would degrade the agent the mesh
actually talks to.

## 4. t2 spike — 27B serves on nightly (live, 2026-07-01): **GO**

Standalone spike run in an **authorized maintenance window** — no in-repo change,
but the live primary was stopped to free memory on the saturated GB10 (see the
memory note below), then restarted on its `26.04` image. Container:
`vllm/vllm-openai:nightly` (already local, no pull) = vLLM
**0.23.1rc1.dev672+g93d8f834d**, free host port 8100, replicating the primary's
serving flags at a trimmed `--max-model-len 8192` / `--gpu-memory-utilization
0.40`.

**Verdict: GO.** The 27B `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` loads and
serves on nightly with MTP active — validating the plan's load-bearing
assumption `c22` and resolving risk `r1` (the deprecated `qwen3_5_mtp` method /
Qwen3.5 hybrid-attention path did **not** break on nightly).

| Check | Result on nightly (0.23.1rc1.dev672) |
|---|---|
| Architecture | ✅ `Qwen3_5ForConditionalGeneration` + draft `Qwen3_5MTP` resolve |
| MTP method | ✅ `qwen3_5_mtp` logged deprecated → **auto-mapped to `mtp`**; `SpeculativeConfig(method='mtp', num_spec_tokens=3)` — MTP stays on, no caller change needed (r1) |
| Quantization | ✅ `--quantization modelopt` → resolved `modelopt_fp4`, served via `FlashInferCutlassNvFp4LinearKernel` (native NVFP4 cutlass) |
| Hybrid attention | ✅ `Using Triton/FLA GDN prefill kernel` — Qwen3.5 Gated-DeltaNet/linear-attention path active, no gibber |
| Correctness | ✅ `17 × 23 = 391` (finish=stop) |
| Tool call (qwen3_coder) | ✅ `get_weather({"city": "Paris"})`, finish=`tool_calls` |
| MTP draft acceptance | ✅ **69.9 %** (174 accepted / 249 drafted; 83 drafts — small sample, at/near the 72–79 % 0.19.0 baseline) |
| Throughput | **~20.7 tok/s** incl. prefill (256 tok in 12.4 s) — parity with the ~18.7–19.1 tok/s 0.19.0 decode baseline |
| Load | 18.65 GiB weights + 25.68 GiB KV (224,824-token cache @ 8192, 27.4× concurrency); init 141 s (compile 39 s) |

**Caveats (do not change the GO):** the 69.9 % acceptance is a single small-sample
probe (83 drafts) that lands just under the 72 % floor but within noise (nightly
also warns `num_speculative_tokens > 1 … may result in lower acceptance rate`);
the throughput includes prefill at a trimmed 8192 / util 0.40, so it is a
**functional-parity** signal, not a production-scale number — **t6** does the
apples-to-apples measurement. MTP works, correctness/tool-calling hold, quant +
hybrid kernels are native.

### Maintenance-window / memory note (risk `r2`, **confirmed**)

At spike time the GB10 was **memory-saturated**: 118/121 GiB used, swap 15/15 GiB
full, ~3 GiB free — the running fleet (primary at `util 0.6` + `max-model-len
262144`, plus minor/embed/rerank + the non-vLLM services) left no room for a
co-resident 27B even at low util. So t2 required **stopping the live primary**
(freed ~72 GiB → 62 GiB available) for the run, then restoring it. This is risk
`r2` confirmed: the fleet cannot co-reside a *second* large gear on nightly — so
**t6's head-to-head will likely need the sequential/standalone method**, and t4's
flip must mind the util budget. (The nightly image was already local; no pull.)

## Scope note

Sections §1–§3 are **before-state only** (t1) — no default flipped, no
`docker-compose.yml` / `env.example` / `catalog.py` change. §4 records the **t2**
live spike verdict (GO). Still pending: t3 (embed/rerank-on-nightly spike), then
t4's actual image-pin flip, t5–t7 Gemma DSpark, t8 trailing gears, t9
shipped-state docs.
