# Qwen3-Embedding-4B — the `embed-deep` slot (2560-dim)

> One entry in lobes's **supported catalog** (`lobes overview --list`). For
> the catalog-vs-warm distinction — what you *can* load vs. what's loaded *now* —
> see [`gateway-fleet.md`](gateway-fleet.md#supported-catalog-vs-warm-backends).
>
> **Status: `configured` — DECLARED, NOT VALIDATED.** No physical boot has run
> this gear on any card in the support table. Every number below that is not
> from the upstream model card is a *derivation*, flagged as such. Per the #108
> rule, nothing here may be cited as load-tested until an acceptance transcript
> lands under `docs/evidence/`.

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

`EMBED_DEEP_GPU_MEM_UTIL` defaults to **0.11**.

**This is a derivation, not a measurement** (#108): ~8 GiB of bf16 weights, plus
the same pooling cache-block overhead the 0.6B demonstrably needs at 0.06 (0.025
fails there with "No available memory for the cache blocks" — vLLM's pooling
runner reserves a cache-block budget regardless of how small the model is).

Against the shipped budgets, adding the deep slot means:

| shape | before | with `embed-deep` |
|---|---|---|
| default fleet (cortex 0.30 + senses 0.14 + embed 0.06 + rerank 0.06) | 0.56 | **0.67** |
| `spark-lobe` (cortex 0.44 + embed 0.06 + rerank 0.06) | 0.56 | **0.67** |

Verify on the box before committing this to a co-residency budget. The
unified-memory boxes have refused naive budget arithmetic before — both
`spark-lobe`'s 0.44 and `thor-lobe`'s 0.30 are *measured* reclaims that replaced
a computed value vLLM rejected.

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
