# Qwen3-Embedding-4B — the `embed-deep` slot (2560-dim)

> One entry in lobes's **supported catalog** (`lobes overview --list`). For
> the catalog-vs-warm distinction — what you *can* load vs. what's loaded *now* —
> see [`gateway-fleet.md`](gateway-fleet.md#supported-catalog-vs-warm-backends).
>
> **Status: `load-tested` on the DGX Spark GB10 only (2026-07-20).** Booted live
> and measured on `spark-f8a9` alongside the running fleet — transcript:
> [`docs/evidence/2026-07-20-accept-embed-deep-gb10.txt`](evidence/2026-07-20-accept-embed-deep-gb10.txt).
> **sm_110 (Jetson AGX Thor) is UNVALIDATED** and needs a manual attention-backend
> override before it will even work — see the sm_110 section below. Retrieval
> quality (MTEB 69.45) remains the upstream model card's number, not ours.

## What it is

The **deep** half of a two-gear embedding lane. The always-on
[`Qwen3-Embedding-0.6B`](qwen3-embedding-0.6b.md) keeps the latency-sensitive hot
path; this 4B is what you reach for when retrieval fidelity outranks speed.

- 4B **dense** text embedding model from the Qwen3 family.
- **2560-dim** native output with **Matryoshka (MRL)** nesting at
  32 / 64 / 128 / 256 / 512 / 768 / 1024 / 1536 / 2048 / 2560 dimensions.
- **32K native** context, served at `--max-model-len 8192`.
- Served via vLLM's `/v1/embeddings` endpoint in pooling mode
  (`--runner pooling --convert embed`), same as the 0.6B.
- No tool parser, no quantization — a pooling model, not a chat model.
- **Served name == catalog id:** `Qwen/Qwen3-Embedding-4B`.
- **MTEB multilingual mean 69.45**, vs the 0.6B's ~64.3 (upstream model card).

### Why the slot is named `embed-deep`, not `embed-4b`

The alias names the *job* — the deeper, higher-resolution pass — not the
checkpoint filling it. Swapping this slot to a larger model later (an 8B, say)
keeps every call site valid. It does **not** keep the index; see below.

## The one rule: the two embedders are not interchangeable

**Embeddings from the 0.6B and the 4B live in different vector spaces.** A
corpus indexed with one can only be queried with the same one. Mixing them does
not raise an error — it silently returns meaningless similarity scores, which is
strictly worse than a failure.

Three consequences worth internalising before adopting this gear:

1. **Adopting the deep slot for an existing index means a full re-embed of that
   corpus.** There is no migration shortcut.
2. **Matryoshka truncation does not bridge the gap.** Truncating the 4B to 1024
   dims produces a 1024-vector that is *not* comparable to the 0.6B's 1024. Same
   width, different space.
3. **Swapping the slot's checkpoint later invalidates whatever it indexed.** The
   alias survives the swap; the vectors do not.

The safe framing is **shallow-recall vs deep-analysis over separate corpora or
separate passes** — never one model indexing and the other querying.

Because of this, the gateway deliberately gives `embed-deep` **no fallback**. If
the deep gear is not wired, the alias does not exist and a request for it is an
unknown model. It never degrades to the 0.6B, because a silent downgrade here
would answer from the wrong vector space — the half-honest posture issue #92
forbids. (This is why `embed-deep` is not a generate-lane capability tier:
`tier_aliases` falls back *upward*, which is right for generation and wrong for
embeddings.)

## Serving

Opt-in on both axes — the container is profile-gated **and** the gateway route
is `*_BASE_URL`-gated, so an existing deployment is byte-identical until you
turn both on:

```bash
# 1. un-gate the container
#    (or add "embed-deep" to COMPOSE_PROFILES in .env)
docker compose --profile embed-deep up -d

# 2. wire the gateway route — in .env:
EMBED_DEEP_BASE_URL=http://vllm-embed-deep:8000
```

Then it answers to the `embed-deep` alias:

```bash
curl -s localhost:8001/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "embed-deep", "input": ["hello world"]}'
```

The literal served id (`Qwen/Qwen3-Embedding-4B`) works too — the gateway's
task-family routing matches on served name and is already generic over multiple
`task="embed"` backends.

To serve it *solo* for isolated testing,
`lobes switch Qwen/Qwen3-Embedding-4B` (task auto-detected from the catalog)
prints the exact compose edits to apply.

## Budget

`EMBED_DEEP_GPU_MEM_UTIL` defaults to **0.11** — **measured** on the GB10
(2026-07-20), not derived. It boots and serves correctly at that value while
co-resident with the full `spark-lobe` fleet:

| measurement | value |
|---|---|
| model weights | 7.56 GiB |
| available KV cache | 11.34 GiB |
| GPU KV cache size | 82,592 tokens |
| CUDA graph pool | 0.84 GiB actual (0.3 estimated) |
| init engine | 23.11 s (compilation 14.69 s) |
| host memory delta | ~+15 GiB |

Against the shipped budgets, adding the deep slot means:

| shape | before | with `embed-deep` |
|---|---|---|
| default fleet (cortex 0.30 + senses 0.14 + embed 0.06 + rerank 0.06) | 0.56 | **0.67** |
| `spark-lobe` (cortex 0.44 + embed 0.06 + rerank 0.06) | 0.56 | **0.67** |

**Do not trust that arithmetic.** vLLM's own numbers don't reconcile with it:
util `0.11 × 121.69 GiB = 13.39 GiB` of budget, yet it allocated
`7.56 + 11.34 + 0.84 = 19.74 GiB`. This is the same unified-memory behaviour that
forced `spark-lobe`'s 0.44 and `thor-lobe`'s 0.30 to be *measured* reclaims
rather than computed ones. The util knob on this card is empirical — the table
above is a bookkeeping convention, not a prediction.

One caveat worth acting on: the shipped service sets
`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` (matching `vllm-embed`), and vLLM
warns that CUDA graph memory is then **not accounted for** during KV allocation,
suggesting 0.1125 if the profiler is re-enabled. The measured graph pool is
0.84 GiB. 0.11 is kept because it demonstrably works here — but an operator
running the deep gear under memory pressure should watch for OOM rather than
trust the headroom.

## Latency — the reason the 0.6B keeps the hot path

Single-stream, n=10 after 2 warmups, GB10, 2026-07-20:

| gear | median | min / max |
|---|---|---|
| `embed-deep` (4B, 2560d, direct) | **42.4 ms** | 41.4 / 44.1 |
| `embedder` (0.6B, 1024d, via gateway) | **11.5 ms** | 10.4 / 12.5 |

**3.69× slower — and that is a lower bound**, because the 0.6B figure includes a
gateway hop the 4B figure does not. This is the whole argument for keeping two
gears rather than swapping one: at ~4× the latency on a path hit by every ingest
and every recall, the deep gear is not a drop-in upgrade.

## sm_110 (Thor): you must set the attention backend by hand

`EMBED_DEEP_ATTENTION_BACKEND` defaults to `auto`, which on sm_110 resolves to
the **broken FLASH_ATTN pooling path** — the one that hangs the `embedder`'s
forward pass while `/health` stays green (#105).

The `SM_110` trait that fixes this for `embedder`/`reranker` keys its knobs by
**profile role name** (`lobes/machines/_traits.py:22,28`). `embed-deep` is a
gear, not a role — outside `lobes.profiles.schema.ROLES` — so **no trait can
reach it**. On Thor, or any future sm_110 board, set it explicitly:

```bash
EMBED_DEEP_ATTENTION_BACKEND=TRITON_ATTN
```

This is the known cost of the gear-vs-role choice. Every opt-in gear shares it;
`embed-deep` is the first where the consequence is a silent hang rather than a
merely suboptimal knob.

**UNVALIDATED on sm_110.** No Thor has booted this gear — the `TRITON_ATTN`
workaround is *inferred* from the embedder's measured sm_110 behaviour, not
measured for this one (#108). Any measurement in this doc taken on the Spark
rides FLASH_ATTN and does not transfer to Thor.

## Relationship to the `embedder` role

`embed-deep` is a **gear**, not an eighth Colleague role. It shares the
`embedder` role's responsibility contract exactly — the two differ in cost and
fidelity, which is what this codebase calls a tier, not a role (the same reason
the 4B `minor` generate gear is not a role). Accordingly it has no entry in
`FEASIBLE_ENV`, no `*_PEER_ORIGIN` / `*_PEER_PROXY` knobs, and no per-machine
`RoleProfile` — it is out of scope for `lobes.profiles.schema.ROLES`, like every
other opt-in gear.

`lobes capabilities` therefore continues to report one `embedder` role backed by
the 0.6B. The deep gear surfaces on `GET /v1/models` when wired.

## See also

- [`qwen3-embedding-0.6b.md`](qwen3-embedding-0.6b.md) — the hot-path gear
- [`gateway-fleet.md`](gateway-fleet.md) — backend wiring and alias resolution
- [`colleague-stack.md`](colleague-stack.md) — the seven-role contract
