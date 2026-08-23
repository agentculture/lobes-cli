# unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M — the llama.cpp gear

> **SUPERSEDED NUMBERS BELOW — read the "MAXN correction" section at the end
> first.** Every throughput figure in the "Measured performance" section was taken
> under `MODE_30W` with the GPU capped at 612 MHz. At MAXN the same lane measures
> **8.46 tok/s decode and 253.84 tok/s prefill** — 3.2x and 3.9x higher — and all
> five acceptance criteria pass.
>
> **SUPERSEDED NUMBERS BELOW - read section 14 of the evidence transcript first.**
> Every throughput figure in "Measured performance" was taken under `MODE_30W`
> with the GPU capped at 612 MHz. At **MAXN** the same lane measures
> **8.46 tok/s decode / 253.84 tok/s prefill** - 3.2x and 3.9x higher - and
> **all five acceptance criteria pass**. See "MAXN correction" at the end.

**Status: `configured`, NOT validated.** The facts below are read off the
downloaded GGUF file and off `llama-server`'s own flag surface. **No acceptance
transcript exists**, so nothing here — and nothing anywhere else in the repo —
may claim this lane VALIDATED (#108). The covering plan
(`docs/plans/2026-08-22-qwen3-8-gguf-llamacpp.md`) lands the evidence in task
t10; this doc gets its measured numbers then.

This is the catalog's **first non-vLLM gear**. It exists to record the engine
axis (`lobes/catalog.py`'s `engine` field, plan task t3) against a real
checkpoint, not to re-point any role: `role_hint` is `candidate`, so no tier
alias resolves to it and the fleet's `cortex` primary is unchanged.

## The checkpoint

Measured off the downloaded file, 2026-08-23, on the Jetson AGX Orin 64 GB this
gear is aimed at:

| fact | value |
|---|---|
| `general.architecture` | `qwen35` |
| parameters | 27.32B |
| file size | 15.32 GiB |
| quantization | `Q4_K_M` (Unsloth Dynamic 2.0, `UD-Q4_K_M`) |
| native context | 262144 (256K), the GGUF's own declared ceiling |

It is the same upstream checkpoint the fleet's vLLM `cortex` primary serves
(`unsloth/Qwen3.8-27B-NVFP4` — see `docs/qwen3.8-27b-nvfp4.md`), in a different
container format for a different engine.

## The engine — why the vLLM fields are empty

Served by `llama-server`, which speaks the OpenAI-compatible API, from the
digest-pinned image the lane carries (see "The lane" below). The catalog entry leaves every vLLM-only
field empty **on purpose**:

- **`tool_parser=""`** — `llama-server` has no `--tool-call-parser` and no
  `--reasoning-parser`. It takes `--jinja` and uses the model's own embedded
  chat template, auto-detecting the tool-call and reasoning shapes
  (`--reasoning-format auto`, `--chat-template-kwargs`). `infer_parser` answers
  `qwen3_coder` for any `qwen3.8` id; that answer names a **vLLM flag that does
  not exist here**, so it is deliberately not carried. `lobes switch` skips it
  too (`serves_with_vllm`).
- **`quantization=""`** — `Q4_K_M` lives inside the `.gguf` file. There is no
  flag to pass.
- **`speculative_config=""`** — llama.cpp **ignores** the checkpoint's MTP head.
  The NVFP4 sibling's self-hosted MTP draft does not run on this engine, so this
  gear has **no speculative decoding**. Declaring the sibling's config would
  advertise a capability the lane cannot serve.
- **`moe_backend=""` / `hf_overrides=""`** — vLLM serve flags; llama.cpp has
  neither surface.

## Feature gaps versus the vLLM cortex

Recorded here so no caller infers parity from the shared checkpoint id. Each is
a **declared** gap from the engine's flag surface, to be re-checked against the
plan's live transcript:

| feature | vLLM cortex | this lane |
|---|---|---|
| MTP speculative decoding | self-hosted draft, measured | **ignored by the engine** |
| ViT (image / video intake) | yes | not served on this lane |
| `preserve_thinking` (#93) | `--default-chat-template-kwargs` | UNVERIFIED |
| strict tools / xgrammar (colleague#320) | armed | UNVERIFIED |

## The lane (plan t4)

`llamacpp-primary` in `lobes/templates/fleet/docker-compose.yml` — the fleet's
first non-vLLM generate service. Properties worth knowing before you touch it:

| property | value | why |
|---|---|---|
| image | `ghcr.io/nvidia-ai-iot/llama_cpp@sha256:f7c67c10…` | Pinned by digest, never a tag. llama.cpp build `38406d597` (10373) on CUDA 13, arm64, `CUDAARCHS=87;110`. |
| gating | `profiles: [llamacpp]` | Opt-in like `vllm-muse`/`vllm-worker`/`vllm-minor` — no existing deployment starts it. |
| exposure | `expose: "8000"`, **no `ports:`** | Reachable only on the compose network, matching the `vllm-multimodal` precedent. The gateway origin is the only way in. |
| GPU request | the shipped `deploy.resources` stanza | A csv-mode board rewrites it to `runtime: nvidia` through its card profile's `gpu_access = "runtime"` declaration and the generated `docker-compose.gpu.yml` — see below. |
| model | `/models/${LLAMACPP_GGUF}`, mounted **read-only** | The lane never downloads. The served file is the one the operator placed. |
| health | `curl -f http://localhost:8000/health` | `llama-server` answers `{"status":"ok"}` once loaded; `start_period` covers the measured 84 s cold load. |

**The image was chosen by measurement, 2026-08-23 on the Jetson AGX Orin 64GB**
(`llama-bench -p 512 -n 128 -ngl 99`):

| build | CUDA | flash attn | pp512 | tg128 |
|---|---|---|---|---|
| `ghcr.io/nvidia-ai-iot/llama_cpp` 10373 | 13 | on | **64.79** | **2.61** |
| `ghcr.io/nvidia-ai-iot/llama_cpp` 10373 | 13 | off | 63.56 | 2.57 |
| `ghcr.io/ggml-org/llama.cpp` 10573 | 12 | on | 63.10 | 2.53 |
| `ghcr.io/ggml-org/llama.cpp` 10573 | 12 | off | 61.98 | 2.52 |

The ggml-org image is the recorded **runner-up**, not a drop-in: swapping to it
is a deliberate act that should be re-measured.

**Serve flags** are the measured-best set: `-ngl 99 -c 262144 --jinja -np 1
-fa on`. They are written as separate `--flag value` list items because
llama.cpp's own arg parser **rejects** the `--flag=value` form every vLLM lane
uses (verified against this image: `error: invalid argument: --ctx-size=4096`).

**No vLLM flag appears in it** — no `--quantization`, no
`--gpu-memory-utilization`, no `--tool-call-parser`, no `--reasoning-parser`,
no `--speculative-config`. Those surfaces do not exist on this engine, and a
translated-looking flag would be a claim the lane cannot honour.

## Rendering it (plan t5)

Nothing about the lane is hand-configured. The `orin` card profile declares
`cortex` on this gear, and the engine follows from the catalog entry for that
model id:

```bash
lobes init --profile orin --shape orin-cortex --apply
```

renders, from repo data alone:

- `.env` — `PRIMARY_MODEL` / `PRIMARY_SERVED_NAME` = this gear,
  `PRIMARY_MAX_MODEL_LEN=262144`, `PRIMARY_URL=http://llamacpp-primary:8000`,
  `COMPOSE_PROFILES=llamacpp`, `MULTIMODAL_FEASIBLE=false`. No
  `PRIMARY_GPU_MEM_UTIL` — this engine has no such flag, so declaring one would
  be a dead knob;
- `docker-compose.shape.yml` — parks `vllm-primary` **and** `vllm-multimodal` in
  the inert `shape-dropped` compose profile. Parking the vLLM cortex lane while
  cortex is *hosted* is the engine swap: both lanes running at once on a
  61.3 GiB board is the failure that prevents;
- `docker-compose.gpu.yml` — `!reset`s every GPU service's `deploy:` stanza and
  sets `runtime: nvidia`, because this board's NVIDIA container toolkit resolves
  to legacy **csv** mode and refuses the `deploy.resources` form at container
  create. `llamacpp-primary` is in that list.

The caller-facing contract does not move: `model=cortex|main|hard` still routes
to `cortex`, and the gateway needed no code change to reach a different engine.

## Memory, measured

At the full 262144 window on the Orin (2026-08-23):

| term | value |
|---|---|
| weights | 15.33 GiB |
| KV cache | 16.00 GiB (**64 KiB/token**) |
| resident total | **~33 GiB** (predicted 31.33 from the GGUF header) |
| load time | 84 s cold, 15.6 s warm-cache |

KV is cheap because only **16 of the 65 layers** hold a per-token cache
(`full_attention_interval = 4`), and those use GQA at `head_count_kv = 4`,
`key_length = value_length = 256`. The other 49 are Mamba/SSM with constant
state. `-c` is the only memory dial this engine has.

That footprint is why `orin-cortex` drops `senses`: ~33 GiB plus senses' ~27.6
GiB does not fit in 61.3 GiB, and the board has **zero swap**.

## `lobes switch` and this gear

`lobes switch` configures the **vLLM** lane. Pointing it at this gear prints an
engine notice and, under `--apply`, writes `.env` **without restarting the
container** — the same protection every "needs a compose edit" gear gets, so a
healthy deployment is never taken down by a switch that could not have worked.
The llama.cpp lane itself is rendered from repo data (compose template + machine
profile + deployment shape) by the plan's tasks t4/t5, not from `VLLM_*` keys.

## Measured performance (plan task t10, evidence `docs/evidence/2026-08-23-spike-qwen38-gguf-llamacpp-orin.txt`)

**Status: functional GO / throughput FAIL-AS-SPECIFIED.** Measured on a Jetson
AGX Orin 64 GB (sm_87, L4T R39.2 / JetPack 7.x, driver 595.78), `MODE_30W` with
clocks pinned via `jetson_clocks` (GPU 612 MHz, EMC 3199 MHz), senses stopped.

### The selected configuration

```text
image  ghcr.io/nvidia-ai-iot/llama_cpp@sha256:f7c67c102b08252e963f9e5f92c3a36554c8f69305eb7ea257c6cd12e24c3191
flags  -ngl 99 -c 262144 --jinja -np 1 -fa on
```

This is the fastest of **five** measured configurations, not a default:

| build | CUDA | fa | quant | pp512 (t/s) | tg128 (t/s) |
|---|---|---|---|---|---|
| ggml-org 10573 | 12 | 1 | Q4_K_M | 63.10 | 2.53 |
| ggml-org 10573 | 12 | 0 | Q4_K_M | 61.98 | 2.52 |
| **nvidia 10373** | **13** | **1** | **Q4_K_M** | **64.79** | **2.61** |
| nvidia 10373 | 13 | 0 | Q4_K_M | 63.56 | 2.57 |
| nvidia 10373 | 13 | 1 | Q3_K_XL | 63.55 | 2.46 |

Note the CUDA-13 image carries an **older** llama.cpp (10373 < 10573), so its win
is the CUDA generation, not newer llama.cpp code. Note also that the **smaller**
quant is **slower** — see "Why it is 2.6 tok/s".

### Latency and depth

| prompt depth | TTFT | decode |
|---|---|---|
| 0 | 1.6 s | 2.61 tok/s |
| 512 | 12.4 s | 2.62 tok/s |
| 2 048 | 41.6 s | 2.62 tok/s |
| 8 192 | 143 s | 2.58 tok/s |
| 32 768 | 610 s (10.2 min) | 2.43 tok/s |

Sustained prefill is 56-63 tok/s and both curves degrade **gently** with depth
(decode -7%, prefill -10% across the measured range). There is no cliff, so:

> **TTFT is a near-linear function of prompt size: `TTFT_seconds ~= depth / 57`.**

Extrapolated to the full window that is ~77 minutes to first token at 262144.
The window is genuinely served, but on this hardware it is a **batch capability,
not an interactive one**. Interactive use is bounded at roughly the low thousands
of tokens. This caveat must travel with every claim about the context length.

### Why it is 2.6 tok/s — seven levers, all measured

| lever | effect |
|---|---|
| long context (8192 vs 262144) | **null** — identical |
| GPU core clock (306 -> 612 MHz) | **null** — exactly 1.00x |
| memory / EMC clock | n/a — already pinned at max |
| CPU saturation | **null** — no core above 54% |
| flash attention | +1.5% |
| CUDA generation (12 -> 13) | +3.2% |
| GGUF quant (Q4_K_M -> Q3_K_XL) | **-6%** — smaller is *slower* |

`GR3D_FREQ` sits at 99-100% throughout yet decode is completely insensitive to a
2x core-clock change: that is **launch-latency bound**, not compute bound
(`GR3D_FREQ` on Tegra measures work *submitted*, not ALU occupancy). And a 20%
smaller model running 6% slower rules out weight streaming — if decode were
bandwidth bound, fewer bytes would mean more tokens/s.

**The deficit against the plan's >=5 tok/s gate is structural**, not a
misconfiguration: the Gated-DeltaNet path is launch-latency bound on sm_87.

For context, the Jetson AGX Thor serves this same checkpoint (NVFP4, vLLM) at
12.1 tok/s. Proxying `cortex` to a peer is ~4.6x faster than serving it locally
here. **The value of a local Orin cortex is independence, not speed.**

## Feature parity vs the vLLM cortex lane

| feature | this lane | note |
|---|---|---|
| reasoning_content | **PASS** | on default flags; `<think>` does not leak into content |
| preserve_thinking (#93) | **PASS** | the GGUF template preserves *by default* — better than the vLLM lane, which needs an explicit flag |
| tool calling | **PASS** | `tool_calls` non-null, `finish_reason: "tool_calls"`; llama.cpp ships a `Qwen 3 Coder` parser for this XML dialect |
| MTP / speculative decoding | **ABSENT** | llama.cpp ignores `blk.64.nextn.*`; ~0.9 GiB of the file is dead weight |
| vision / ViT | **ABSENT** | text-only GGUF; no mmproj companion |

## MAXN correction — the numbers above are a floor, not a ceiling

Everything in "Measured performance" was taken under `MODE_30W` with the GPU
capped at 612 MHz. The board's real figures, measured at MAXN
(`sudo nvpmodel -m 0` + reboot + `sudo jetson_clocks`, GPU pinned 1300.5 MHz):

| metric | 612 MHz (MODE_30W) | **MAXN 1300.5 MHz** | gain |
|---|---|---|---|
| pp512 prefill | 64.79 | **253.84 ± 1.45** | **3.92x** |
| tg128 decode | 2.61 | **8.46 ± 0.00** | **3.24x** |
| sustained decode (900 tok) | — | **8.43** | — |

Clock scaling, measured by pinning `min=max` at each point:

| GPU clock | decode |
|---|---|
| 306 MHz | 1.36 tok/s |
| 612 MHz | 2.61 tok/s |
| 1300.5 MHz | 8.46 tok/s |

The gain is **superlinear** against the 2.12x clock step because MAXN also
restores 4 disabled CPU cores (12 online, was 8) and lifts the power budget.

### What this changes

- **All five acceptance criteria pass.** The verdict is **FULL GO**, not the
  split result the earlier sections describe.
- **TTFT rule:** `TTFT_seconds ~= depth / 254` (was `/ 57`). A full 262144-token
  prefill is ~17 minutes, not ~77.
- **Decode is now bandwidth-bound, the healthy regime:** 16.45 GB of weights at
  8.46 tok/s = 139 GB/s = **68% of Orin's 204.8 GB/s peak**. At 612 MHz it was
  21% — which is what made the lane look pathological. **The Gated-DeltaNet path
  was never broken; it was starved by the power cap.**
- **Against the Thor** (12.1 tok/s on NVFP4/vLLM) the local lane is **1.4x
  slower, not 4.6x**. The earlier "independence, not speed" framing understated
  it — the lane is competitive.

### Operational requirement

**This lane requires MAXN.** On a stock `MODE_30W` Orin it delivers 2.6 tok/s and
misses the >=5 tok/s bar. `sudo nvpmodel -m 0` needs a reboot; pair it with
`sudo jetson_clocks`.

Thermals were verified rather than assumed — 100 s of sustained generation:

| t | tj | fan | GPU clock |
|---|---|---|---|
| 20 s | 64.9 C | 42% | 1300 MHz |
| 60 s | 68.7 C | 49% | 1300 MHz |
| 100 s | 70.9 C | 52% | 1300 MHz |

`nvfancontrol` auto-ramped and the GPU **held 1300 MHz with no throttling**,
~25 C below the ~95 C throttle point. **No manual fan lock is needed.** The signal
that would change that: `cur_freq` dropping below 1300 MHz under load.

### Why the earlier sections got it wrong

The original clock test reported a null (306 -> 612 MHz = 1.00x). It was an
artifact: `cur_freq` was read **at idle** while the `nvhost_podgov` governor was
active at `min=306 / max=612`; under load the governor boosted to 612, so the
test compared 612 against 612. Pinning `min=max` before attributing anything to
clock is the correct instrument. See ledger deviation **d8**.

## The quant ladder — measured on BOTH axes

> **Interim decision: `UD-Q4_K_M`.** It is the measured knee from *both*
> directions. The measurement that would actually settle it — error rate on
> cortex-shaped tasks — is deliberately deferred to **issue #194**. Nothing
> below claims the quality question is closed.

Speed: `llama-bench -p 512 -n 128 -ngl 99 -fa 1`. Quality: `llama-perplexity`,
wikitext-2 test set, 200 chunks @ 512 ctx (~102K tokens), identical corpus and
chunk count per quant — a *paired* comparison. Every run under enforced-identical
conditions (MAXN, GPU pinned 1300.5 MHz, the same four containers, **zero active
downloads**), recorded into each log per `docs/measuring-lane-performance.md`.

| quant | size | pp512 | tg128 | PPL | Δ speed | Δ quality | 500-tok answer |
|---|---|---|---|---|---|---|---|
| UD-Q3_K_XL | 12.23 GiB | — | 8.56 | 6.7922 ± 0.074 | +1.2% | -1.20% | 58.4 s |
| **UD-Q4_K_M** | 15.33 GiB | 253.46 | **8.46** | 6.7118 ± 0.073 | +0.0% | +0.00% | 59.1 s |
| UD-Q5_K_M | 18.41 GiB | 232.36 | 7.15 | 6.6970 ± 0.073 | -15.5% | +0.22% | 69.9 s |
| UD-Q6_K | 20.46 GiB | 225.02 | 6.56 | 6.6857 ± 0.072 | -22.5% | +0.39% | 76.2 s |

### Read it as seconds paid, not percentages

A percentage makes 0.39% sound like a rounding error and 22.5% sound like a
catastrophe. Neither framing helps. What a caller experiences is **time**:

| from Q4_K_M | extra seconds per 500-token answer | perplexity bought |
|---|---|---|
| → Q5_K_M | **+10.8 s** | +0.22% |
| → Q6_K | **+17.1 s** | +0.39% |
| → Q3_K_XL | −0.7 s | **−1.20%** (worse) |

**Two full quantization levels above `Q4_K_M` buy 0.39% perplexity for 17 seconds
on every answer.** Descending to `Q3_K_XL` saves 0.7 s and costs 1.20% — the
largest quality delta in the set, in the wrong direction.

### Why size does not explain the speed curve

| step | size change | speed change |
|---|---|---|
| Q3 → Q4 | **+25.3%** | **−1.2%** |
| Q4 → Q5 | +20.1% | **−15.5%** |
| Q5 → Q6 | +11.1% | −8.3% |

A 25% size increase costs 1.2%, while a 20% increase costs 15.5%. Size is not the
driver — the **4-bit → 5-bit dequantisation kernel boundary** is where the cost
appears, and each further bit adds unpacking work per weight. An earlier note in
this repo extrapolated a size law from the Q3→Q4 pair and predicted Q5_K_M at
~8.38 tok/s; it measured **7.15**. Do not extrapolate across quant families —
measure each rung.

### What perplexity does NOT tell you

Perplexity is a **log measure of next-token prediction on wikitext**. It is **not
a decision-error rate**, and the mapping from PPL to task accuracy is neither
linear nor established for this checkpoint. The 0.22% Q4→Q5 gap is **0.20× the
confidence half-width** — real in direction, statistically indistinguishable from
zero in magnitude. It could correspond to no behavioural difference at all.

This matters more for `cortex` than for any other role: it is the fleet's
reasoning / deciding / **final-authority** lobe, and the cost of a wrong call
there is whatever that decision propagates into — not a percentage of a
perplexity score. **The risk side of this trade is unmeasured, not small.**
Issue #194 tracks measuring it properly.
