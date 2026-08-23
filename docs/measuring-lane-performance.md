# Measuring a lane's performance — the five axes

> **Why this document exists.** A single throughput number is not a description
> of a lane. During the Qwen3.8 GGUF bring-up on the Jetson AGX Orin
> (2026-08-23) two headline figures each turned out to be misleading in a
> different way, and both mistakes were caught only by measuring a *curve*
> instead of a *point*. This records the method so the next model — and the
> existing lanes on other boxes — get measured the same way.

Tooling: **`scripts/prefill-depth-curve.py`** (stdlib only, same convention as
`lobes/assess.py`). It speaks the OpenAI streaming API, so it works against any
lane — vLLM, llama.cpp, a peer's gateway — without engine-specific code.

## The five axes

| # | Axis | What it answers | Why a single number lies |
|---|---|---|---|
| 1 | **Prefill, overall** | how fast a short prompt is ingested | fixed per-request cost dominates; flatters the lane |
| 2 | **Prefill vs depth** | how ingestion decays as the prompt grows | attention cost accumulates; only visible deep |
| 3 | **TTFT** | what a caller actually waits for | derives from 1+2; the number users feel |
| 4 | **Decode vs prompt depth** | does a long prompt slow generation | KV grows; usually mild, but not zero |
| 5 | **Decode vs generation length** | the **sane maximum** answer length | KV also grows *while generating* |

Axes 2 and 5 are the ones normally skipped, and they are where the surprises live.

## Run it

```bash
# axes 1-4: the depth curve
python3 scripts/prefill-depth-curve.py --url http://127.0.0.1:8090 \
    --model cortex --depths 0,512,2048,8192,32768

# axis 5: the sane-max generation sweep
python3 scripts/prefill-depth-curve.py --url http://127.0.0.1:8090 \
    --model cortex --depths 0 --gen-sweep 4096 --gen-sample-every 256

# through a lobes gateway with the auth gate armed
python3 scripts/prefill-depth-curve.py --url http://127.0.0.1:8000 \
    --model cortex --api-key "$LOBES_KEY" --json >> docs/evidence/<transcript>.txt
```

Deep depths are slow by construction — a 100K-token prefill is minutes. Start
shallow, add depths as the curve reveals itself, and use `--max-seconds` so one
hung depth cannot stall a run.

## Two rules the Orin bring-up produced the hard way

**1. Never quote a shallow prefill number as the lane's prefill rate.**
`llama-bench -p 512` measured **254 tok/s**. The same lane at 115K tokens ran at
**129 tok/s** — a 49% overstatement. The decay flattens rather than collapsing
(98K→115K lost only 7%, versus 42% over the first 98K), so deep context stays
usable at a predictable premium; but a TTFT estimate built on the shallow figure
understates deep prompts by ~2x.

Measured curve (Qwen3.8-27B UD-Q4_K_M, llama.cpp, Orin sm_87 at MAXN):

| depth | instantaneous prefill |
|---|---|
| 2K | 240 tok/s |
| 84K | 148 tok/s |
| 98K | 139 tok/s |
| 115K | 129 tok/s |

**2. Pin the clock before attributing anything to the clock.**
An early conclusion — "GPU clock has no effect, so the deficit is structural" —
was produced by reading `cur_freq` **at idle** while a devfreq governor was
active (`min=306 / max=612`). Under load the governor boosted to 612, so the test
compared 612 against 612 and correctly returned 1.00x for a change that never
happened. Pinning `min=max` at each point gave the real answer:

| GPU clock | decode |
|---|---|
| 306 MHz | 1.36 tok/s |
| 612 MHz | 2.61 tok/s |
| 1300 MHz (MAXN) | 8.46 tok/s |

The lesson generalises past Jetson: **a governed value read at rest does not
describe that value under load.** Pin it, or measure it during the run.

## Recording the result

Emit `--json` and append it to the lane's evidence transcript under
`docs/evidence/`. Always cite the **server's own `prompt_tokens`**, never the
requested depth — the script reports it that way for exactly this reason.

Record the conditions alongside the numbers, because they change the answer by
multiples: power mode and clocks (on Jetson, `nvpmodel -q` and
`/sys/class/devfreq/*/cur_freq`), what else was resident on the box, and the
image digest. A throughput figure without its power mode is not reproducible.
