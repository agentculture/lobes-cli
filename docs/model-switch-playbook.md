# Model-switch playbook — how to swap a served checkpoint without guessing

Written after the 2026-07-31 `cortex` swap (text-only
`sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` →
multimodal `unsloth/Qwen3.6-27B-NVFP4`). Every item below is something that run
either cost time or produced a claim weaker than it should have been. Read this
**before** the next switch, not after.

The point of this page: the next switch should be **better measured** than the
last one. Most of what follows is about ordering — several of these steps are
worthless unless done *before* the old model is gone.

## 1. Benchmark the INCUMBENT first, on today's engine

The single biggest weakness of the 2026-07-31 run. The candidate was measured at
19.0 tok/s sustained decode and compared against the incumbent's *recorded*
18.7–19.1 — but those recorded numbers came from a **different vLLM build**
(0.19.0+nv26.04) at a different `gpu_mem_util`. The comparison was never
controlled, so "level with the incumbent" is a weaker claim than it sounds.

**Do this instead:** before touching anything, run the full benchmark against the
*currently serving* model on the *current* engine. That baseline is unrecoverable
once the model is swapped — re-creating it costs a second full boot cycle
(~13 min of downtime here).

Capture, per case:

| metric | how |
|---|---|
| TTFT | time to first content delta on a streamed request |
| decode tok/s | **`usage.completion_tokens`**, never a count of stream chunks |
| prompt tokens | `usage.prompt_tokens` |
| MTP acceptance | `docker logs <engine> \| grep SpecDecoding` — mean acceptance length, per-position rate, avg draft acceptance |
| KV pool + ceiling | the boot log's `Available KV cache memory` + `GPU KV cache size` (the ceiling is arithmetic — see §8) |

> **Trap:** counting SSE chunks under-reports decode throughput, because
> speculative decoding delivers **multiple tokens per chunk**. The first pass of
> the 2026-07-31 benchmark reported ~8.2 "tok/s" this way; the same run measured
> against `completion_tokens` was **19.0**. More than 2× wrong.

Run at least three shapes — short prompt/short gen, medium, and a long
generation (512+ tokens) — because decode rate climbs with generation length as
fixed overheads amortise (14.9 → 16.4 → 19.0 across those three).

## 2. Know who hardcodes the served id — it WILL break them

**No consumer in this mesh addresses the fleet by role name.** Audited
2026-07-31 across culture/colleague, eidetic, reachy-mini-cli and the lobes
agent's own `culture.yaml`: every one of them puts a **raw served model id** on
the wire. Role names (`cortex`, `main`, `hard`) are supported by the gateway and
used by colleague for `/capabilities` *discovery* — but what it then sends is the
`model` field it read from that response, i.e. the raw id.

So **changing `PRIMARY_SERVED_NAME` 404s every consumer**, instantly and
silently:

```text
sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP  ->  HTTP 404   <- what everyone sends
unsloth/Qwen3.6-27B-NVFP4                 ->  HTTP 200
cortex / main / hard                      ->  HTTP 200
```

Known hardcoders to check and update (non-exhaustive — re-audit each time):

- `colleague/colleague/config.py` — `_DEFAULT_MODEL`
- `lobes-cli/culture.yaml` — the agent's own `model: vllm-local/<id>`
- `reachy-mini-cli` — `docs/operating-reachy.md`, `reachy/vision/scene.py`,
  `reachy/speech/llm.py`
- `eidetic-cli` — `eidetic/memory/embed.py` (embed lane; unaffected by a
  *generate* swap but breaks on an embed swap)

Pick one deliberately, before the swap:

1. **Dual served-name** — vLLM's `--served-model-name` accepts multiple values,
   so the lane can answer to both the old and new id. Zero-touch for every
   consumer; the cost is that `/v1/models` advertises a name that is no longer
   accurate.
2. **Migrate consumers to role names** — correct long-term (it is what the role
   layer exists for) and makes future swaps free, but needs a PR per repo.
3. **Accept the break** — fine only when you control every caller and can
   coordinate. This is what the 2026-07-31 run did, deliberately.

## 3. Update `PRIMARY_SERVED_NAME` in `.env`, not just the compose override

The gateway builds its routing table from `PRIMARY_SERVED_NAME`. A compose
override that sets `--served-model-name` on the vLLM side **only** leaves the
gateway rewriting `model=cortex` to a checkpoint the engine no longer serves:

```text
The model `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` does not exist.   (404)
```

…with a perfectly healthy engine behind it. Keep one source of truth: put the id
in `.env` and have the override read `${PRIMARY_MODEL}` / `${PRIMARY_SERVED_NAME}`.

## 4. Field trap: `reasoning` vs `reasoning_content`

This vLLM build returns the thinking trace as **`reasoning`**. A probe reading
`reasoning_content` gets `None` and looks exactly like a model that has stopped
thinking. On 2026-07-31 that nearly produced a false "promotion blocker —
reasoning traces are being discarded" verdict; the giveaway was the token
accounting (1587 `completion_tokens` against ~244 tokens of `content`, so ~1340
tokens were *somewhere*).

**Always reconcile `usage.completion_tokens` against the length of every field
you can see.** If they don't add up, you are reading the wrong field — not
watching a regression. Dump `sorted(message.keys())` before concluding anything.

## 5. Write probes with negative controls, and make them hard

A probe that a model can pass by echoing the prompt proves nothing.

| lane | weak probe | use instead |
|---|---|---|
| image | "what colour is this square?" | still fine as a smoke test **with** an opposite-colour control |
| **video** | "describe this clip" | **directional motion**: same clip forwards and reversed, ask left-to-right vs right-to-left. A single-frame read cannot pass this; only real temporal processing can |
| thinking | "think step by step" on easy arithmetic | a problem with a *misleading* framing (the hotel-missing-dollar puzzle) — easy questions get answered without engaging thinking at all |
| tool calling | any tool call | `strict: true` **with thinking ON** — that exact combination is what broke in colleague#320 |
| preserve_thinking | eyeballing the trace | two-turn `prompt_tokens` delta with and without the trace in history (a real delta of 800 tokens is proof; 0 is not) |

Keep the negative control in the transcript. "red→red, blue→blue" is evidence;
"red→red" alone is not.

## 6. Budgets are measured, never computed

Unchanged from `machine-profiles.md`, restated because it keeps mattering: vLLM
checks free-at-boot ≥ `util × total`, and on a unified-memory card the host OS,
page cache and co-resident stacks share the pool. `thor-muse`'s hypothesised 0.40
was **refused**; 0.55 measured. `thor-worker`'s 0.45 booted first try.

Record both the value that booted **and** any value that was refused — the
refusal is the more useful number next time.

The 2026-07-31 cortex candidate: `gpu_mem_util=0.44` at the full `262144`
window booted first try — KV pool **26.39 GiB / 756,642 tokens** (a ≈2.89×
*ceiling*, see §8), against the incumbent's 888,946 / 3.39× at the same knobs.
The unquantized bf16 ViT costs ≈132,300 tokens of KV pool (~15%).

## 7. Sequence the downtime

Download the new weights **while the old model is still serving**. The
2026-07-31 swap pulled 22 GB (~15 min) with cortex up the whole time, so actual
downtime was the swap alone (~13 min, dominated by weight load + a 262144-token
profiling pass + CUDA graph capture).

Order that works:

1. benchmark the incumbent (§1) — *unrecoverable later*
2. `hf download <new-model>` in the background, old model still serving
3. stage the lane override; `docker compose config` to confirm the resolved
   command before anything restarts
4. update `.env` (§3) and swap
5. re-probe: text, vision, video, thinking, strict tools, preserve_thinking
6. benchmark the candidate, compare against §1 on the *same* engine
7. write the transcript with a split verdict — what it proved AND did not

Keep the rollback one flag wide: drop the `-f <override>` and restore the
commented `.env` lines. Both were left in place on 2026-07-31.

## 8. Concurrency figures are CEILINGS, not measured throughput

`Nx concurrency` anywhere in this repo means `KV pool / max_model_len` — how
many full-context requests the KV cache could *hold*. It is arithmetic off the
boot log, not a serving measurement, and it must never be multiplied by a
single-stream tok/s.

Measured on the `worker` lane by two independent consumers (2026-07-31):

| | advertised | measured |
|---|---|---|
| concurrency | 14.07x ceiling | saturates near width **8-9** |
| per-stream decode | ~50.8 tok/s (single-stream) | **~30 tok/s** at high width |
| aggregate @ width 14 | — | 268.1 tok/s — a 5.5% gain over width 8 for 75% more load |

(embodiment; colleague#361.) So the useful operating point was roughly HALF the
ceiling, and per-stream throughput degraded ~40% getting there.

Rule: quote the ceiling and the measured saturation together, or quote neither.
If concurrency has not been measured for a lane, say so — the ceiling alone
reads as a capacity claim it cannot support.

## 9. What still is not covered

Honest gaps in the current probe set, for whoever does this next:

- **No quality benchmark.** Nothing here measures whether the new model is
  *better* — only that it is alive, fast, and structurally intact. A swap that
  degrades reasoning quality would pass every gate on this page.
- **No concurrency benchmark.** All numbers are single-stream. The KV pool
  implies ~2.89× concurrency at full context; nothing measures behaviour there.
- **No long-context probe.** The lane serves 262144 tokens; the longest probe
  used ~94 prompt tokens.
- **No pressure/shedding test** against the new lane.

## See also

- [`machine-profiles.md`](machine-profiles.md) — measured budgets, per-card knobs
- [`qwen3.6-27b-nvfp4-multimodal.md`](qwen3.6-27b-nvfp4-multimodal.md) — the
  candidate this playbook was written from
- `docs/evidence/` — transcripts, each with a split verdict
