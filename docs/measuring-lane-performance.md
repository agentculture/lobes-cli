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

## Rule 3 — a number without its conditions is not reproducible

The two rules above are both instances of a third, more general one:
**state the conditions a measurement was taken under, or the number cannot be
compared to anything.**

During the Orin bring-up two comparisons were nearly corrupted this way. The
clock artifact above is one (a governed clock read at idle). The other: one arm
of a build-comparison matrix ran while a 13 GB model download was active, and a
27B model lane was still resident in memory when a quant ladder was first queued.
For a comparison whose interesting differences are **1-2%**, that variance is
larger than the signal.

`scripts/prefill-depth-curve.py` reports the server's own `prompt_tokens` for the
same reason: so the depth in the record is measured, not requested.

When benchmarking to compare, **enforce and record**:

- power mode and clocks (Jetson: `nvpmodel -q`,
  `/sys/class/devfreq/17000000.gpu/{cur,min,max}_freq`, `bwmgr/cur_freq`)
- the exact co-resident set (`docker ps`) and any active downloads
- free memory before the run, and thermals before **and** after
- the image digest and the model file

and **re-baseline the incumbent under the same conditions** rather than comparing
a fresh clean run against an older one taken under unknown load. Re-measuring a
number you already have is cheap; a wrong conclusion drawn from mismatched
conditions is not.

## Which measurements need a quiet box, and which do not

Not every metric has the same sensitivity to machine load, and treating them
alike wastes time in one direction and corrupts results in the other.

| Metric | Load-sensitive? | Consequence |
|---|---|---|
| throughput (prefill, decode, TTFT) | **yes** | a concurrent download or a resident model lane changes the number — enforce a quiet box (Rule 3) |
| perplexity / quality | **no** | a deterministic function of (model, corpus); load changes how long it takes to compute, never the value |

So quality sweeps can run **in parallel** with downloads and other I/O, while
speed benchmarks must not. Serialising quality measurements wastes hours;
parallelising speed measurements silently corrupts them.

Record the conditions either way — a reader should not have to know this rule to
trust the number.

## Measuring the quality axis, not just the cost axis

A throughput table alone cannot answer *"how slow is worth the quality?"* — it
measures only the numerator. A claim like *"this quant costs 15% for a quality
step up"* has a **measured** cost and an **assumed** benefit unless the benefit
was measured too.

`llama-perplexity` over a fixed corpus gives the missing axis:

```bash
docker run --rm --runtime nvidia -v "$PWD/models:/models:ro" -v "$PWD:/data:ro" \
  --entrypoint /usr/local/bin/llama-perplexity <image> \
  -m /models/<file>.gguf -f /data/wikitext.raw -ngl 99 -c 512 --chunks 40
```

Use the **canonical wikitext-2 test set** so the numbers compare to published
figures, not only to each other, and hold the chunk count identical across
quants.

**Report it as a perplexity delta, never as a "quality delta".** Perplexity is a
proxy: it correlates with quality but does not capture reasoning or tool-use
degradation, which is what a reasoning lane actually does. Pair it with the
correctness probes rather than substituting for them.

## Recording the result

Emit `--json` and append it to the lane's evidence transcript under
`docs/evidence/`. Always cite the **server's own `prompt_tokens`**, never the
requested depth — the script reports it that way for exactly this reason.

Record the conditions alongside the numbers, because they change the answer by
multiples: power mode and clocks (on Jetson, `nvpmodel -q` and
`/sys/class/devfreq/*/cur_freq`), what else was resident on the box, and the
image digest. A throughput figure without its power mode is not reproducible.
