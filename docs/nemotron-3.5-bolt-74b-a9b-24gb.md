# NVIDIA Nemotron 3.5 Bolt 74B-A9B 24GB — the `worker` role

> One entry in lobes's **supported catalog** (`lobes overview --list`). For the
> catalog-vs-warm distinction — what you *can* load vs. what's loaded *now* —
> see
> [`gateway-fleet.md`](gateway-fleet.md#supported-catalog-vs-warm-backends).
>
> **Status: the `worker` role_hint (operator decision, 2026-10-26)**,
> replacing `nvidia/Qwen3.6-35B-A3B-NVFP4` (now a kept candidate — see
> [`nvidia-qwen3.6-35b-a3b-nvfp4.md`](nvidia-qwen3.6-35b-a3b-nvfp4.md)). The
> worker contract does not move: Bolt is a fast multimodal (image+video)
> coder/doer with tool calls, which is exactly what `roles.py` already
> promises. It is hosted on the **DGX Spark** — the only card family its IQ2
> experts run on — and every other mesh member reaches `worker` through the
> mesh. Measured as a standalone container first
> ([`evidence/2026-10-26-accept-nemotron-bolt-spark.txt`](evidence/2026-10-26-accept-nemotron-bolt-spark.txt)).
>
> **The packaged template cannot express Bolt's flags yet.** The fleet
> `vllm-worker` lane has no knob for `--linear-backend`, the mamba cache
> flags, `--kv-cache-memory-bytes`, the media limits or the reasoning-parser
> plugin, and it always passes `--quantization`. The Spark serves it through a
> full `command:` replacement in its hand-kept `docker-compose.override.yml`
> (the [recipe](#serving-it-on-the-spark) below). `lobes switch` to this id
> will not boot on the stock template either (see [Engine](#engine)).

## What it is

| | |
|---|---|
| id | `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` |
| revision | `c6cc2e8ffa353403213d6b9b18aaf8eb684d705a` |
| architecture | `NemotronHPuzzleForCausalLM` in a `nemotron_h_omni` wrapper: hybrid Mamba-2 / attention with sparse MoE, about 9B active of 74B total |
| modalities | text, image and video (RADIO vision encoder); no audio |
| context | 262144 native (`llm_config.max_position_embeddings`) |
| quantization | `modelopt_mixed` (the checkpoint's `quant_method` is modelopt, `MIXED_PRECISION`): IQ2_XXS on the routed experts (~2.06 bits, ggml packing), Q8_0, and W4A16 NVFP4 on the mixer projections — 24.09 GiB loaded |
| tool parser | `qwen3_coder` (the card's own serve command) |
| reasoning parser | `ultra_v3`, shipped as a plugin file inside the checkpoint |
| speculative decoding | DFlash, 7 draft tokens, from `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr` @ `449d2414…` |
| licence | OpenMDW 1.1 (the card) |

All of the above was read from the checkpoint's own `config.json` and
`hf_quant_config.json` at the pinned revision, and from its card's vLLM
command.

## Engine

- **sm_12x only.** vLLM's IQ2 expert path is gated to Blackwell consumer/GB10
  silicon. The Jetson AGX Orin (sm_87) refuses it on v0.32.0 and on main; the
  Jetson AGX Thor (sm_110) has not been tried.
- **Nightly plus `b12x`.** The card names
  `vllm/vllm-openai:nightly-77154ca6116ae34cbfe314beef1016bde9b63989` and
  installs `b12x` at container start (`--linear-backend b12x --moe-backend
  b12x`). The measured run used that nightly with `b12x==1.5.1` baked into a
  derived image so the boot runs offline (vLLM `0.32.0rc1.dev100+g77154ca61`).

## Serving it on the Spark

The measured recipe is the card's vLLM command with three deviations:

1. **`--kv-cache-memory-bytes 4294967296` (4 GiB).** On the Spark's unified
   memory a KV pool sized by `--gpu-memory-utilization` is not
   deterministic: vLLM's profiling counts other processes' memory growth, so
   the same flag gives very different pools from boot to boot. Pinning the
   pool makes the footprint predictable. 4 GiB holds 350,464 fp8 tokens = 1.34
   full 262144-token requests; `--max-num-seqs` is 1 as on the card.
2. **`--allowed-local-media-path /` dropped.** That flag lets any caller read
   any file on the box through a `file://` URL.
3. **Revisions pinned**, and the server bound to `127.0.0.1`.

The full flag list is in the evidence transcript.

## Measured — DGX Spark GB10, 2026-10-25

From
[`evidence/2026-10-26-accept-nemotron-bolt-spark.txt`](evidence/2026-10-26-accept-nemotron-bolt-spark.txt):

| | |
|---|---|
| weights | 24.09 GiB, 150.8 s to load |
| KV pool | 4.0 GiB pinned = 350,464 tokens (1.34× at 262144) |
| host memory available after the probes | 70,146 MiB |
| known-answer | PASS (`391`, reasoning split off) |
| image | PASS on red / blue / green solid-colour controls (synthetic) |
| tool call | PASS — `get_weather({"city": "Haifa"})`, `finish_reason: tool_calls` |
| speed | 69.3 tok/s wall on one short request (includes prefill and reasoning) |
| long context | PASS — 231,935-token needle retrieved in 200.3 s |

## Measured — as the `worker` lane, 2026-10-25

PART 2 of the same transcript: the Spark's lobes `vllm-worker` lane (cold
compile, the same 350,464-token pinned pool), reached as `model=worker` from
the spark, thor and spark2 gateways — the Thor and spark2 by mesh forward,
with `X-Lobes-Mesh-Member: spark`. Through the Thor's forward the red/blue
image controls and a `get_weather` tool call passed.

## Not proven

- The packaged template serving it (the Spark uses an override command), a
  deployment shape that hosts it, and `lobes switch` to it — switch prints
  the gear's serving note and, on `--apply`, writes `.env` without
  restarting.
- Any card other than the Spark GB10.
- Concurrency above 1, sustained decode under load, and DFlash acceptance
  rates.
- Real-image and video quality (the image probes are synthetic).
