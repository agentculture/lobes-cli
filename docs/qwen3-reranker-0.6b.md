# Qwen3-Reranker-0.6B — reranker gear (score / rerank)

> One entry in lobes's **supported catalog** (`lobes overview --list`). For
> the catalog-vs-warm distinction — what you *can* load vs. what's loaded *now* —
> see [`gateway-fleet.md`](gateway-fleet.md#supported-catalog-vs-warm-backends).
>
> **On vLLM nightly since the fleet-wide nightly-unification migration**
> (`docs/vllm-nightly-migration.md` §3/§5, t3/t4) — this gear now runs the same
> pinned digest as the primary and multimodal gears, not the `nv26.04-py3` /
> `0.19.0` build the 2026-06-19 benchmark below was measured on. The compose
> template pins
> `vllm/vllm-openai@sha256:8bd082c274fae025b7079498fe1da65182ba1d4c2188c0f5a68c1042c38c3695`,
> and the live lane on the DGX Spark reported vLLM `0.26.1rc1.dev942+g5a4c8d992`
> on 2026-08-30
> (`docs/evidence/2026-08-30-accept-reranker-template-spark.txt`). (This page
> previously cited `sha256:7c5a10e9...` / `0.23.1rc1.dev672`; that claim was
> stale and is corrected here — issue #227, decision q2.) The nightly t3 spike (§5)
> re-confirmed `/v1/rerank` (sorted, correct ranking) and `/v1/score` (raw
> pairwise), `--runner pooling --convert classify` unchanged — no
> serving-flag drift (the spike's one hiccup was a memory-profiling race
> during concurrent teardown, not a model/nightly incompatibility). The
> benchmark numbers below remain the historical 0.19.0 record.

## What it is

- 0.6B **dense cross-encoder** from the Qwen3 family — a
  `Qwen3ForSequenceClassification` model with a binary **yes / no** logit head.
- Scores (query, passage) pairs for retrieval re-ranking.
- **32K native** context, served at `--max-model-len 8192` (tiny KV footprint).
- Served via vLLM's pooling/scoring mode (`--runner pooling --convert classify`;
  unchanged since the fleet's move to vLLM nightly — see the note above) — one
  backend handles both `/v1/rerank` (Jina/Cohere shape, sorted best-first) and
  `/v1/score` (raw pairwise scores, input order).
- No tool parser, no quantization flag — this is a scoring model, not a chat model.
- **Served name == catalog id:** `Qwen/Qwen3-Reranker-0.6B`.

## Serving

Served as a **warm fleet backend** alongside the 27B primary and the embedder on
the DGX Spark GB10 (128 GB unified memory). Its small footprint (0.6B weights,
32K KV window) keeps the KV cache tiny so all three backends co-fit.

The warm path is the **fleet** (`lobes init --fleet` then `lobes fleet up --apply`).
To serve it *solo* for testing, `lobes switch Qwen/Qwen3-Reranker-0.6B` (the task is
auto-detected from the catalog) prints the exact compose edits to apply.

Key compose flags:

- `--runner pooling --convert classify` — vLLM scoring mode on this build (replaces
  the old `--task score`; vLLM auto-resolves `--convert auto` to `classify` for a
  `*ForSequenceClassification` arch, but pass it explicitly to silence the notice)
- `--hf-overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}'`
- `--chat-template /usr/local/share/lobes/qwen3_reranker.jinja` — the model
  card's judge prompt (issue #227; see the next section). The file arrives in
  the deployment dir as a scaffold file and is bind-mounted read-only:
  `./qwen3_reranker.jinja:/usr/local/share/lobes/qwen3_reranker.jinja:ro`
- `--max-model-len 8192`
- `--gpu-memory-utilization 0.06`

## Prompt template and calibration

### The mechanism — why the flag is load-bearing

vLLM applies a **score template only when `--chat-template` is passed
explicitly**. The in-source `FIXME` in the served build's scoring
`io_processor.py` (`get_score_prompt`) is deliberate: a tokenizer's own chat
template is *not* trusted for scoring, so nothing is applied implicitly. The
other half of the door is `SupportsScoreTemplate` — and
`Qwen3ForSequenceClassification` does **not** implement it. With no flag the
request therefore fell through to `default_tokenizer_encode`, i.e. a bare
`query` + `document` concatenation: **no system judge line, no
`<Instruct>`/`<Query>`/`<Document>` block, no empty `<think>` block**. Measured
on the DGX Spark, 2026-08-30
(`docs/evidence/2026-08-30-baseline-reranker-untemplated-spark.txt`), that came
to **~17–30 prompt tokens per pair** — provably too few for the ~60-token judge
prompt to be present.

vLLM's own serving line for exactly this `--hf-overrides` invocation says so:
`examples/pooling/score/qwen3_reranker_online.py` appends
`--chat-template examples/pooling/score/template/qwen3_reranker.jinja`. lobes
never did. It does now.

lobes **vendors that file verbatim** (cite-don't-import — never a path inside
the image that a digest bump can move) as
`lobes/templates/fleet/qwen3_reranker.jinja`, sha256
`e1ee98e69aab7b2da366edf1c50efcef37e34b4a0c50fb816336213e68d9047a`. It is
registered in `FLEET_TEMPLATES`, so `lobes init` scaffolds it, `lobes doctor`
reports it under `scaffold_files` when it is missing, and
`lobes doctor --fix --apply` restores it byte-identical.

The flag is **BAKED** into the `vllm-rerank` compose command, hardcoded exactly
like `--hf-overrides`. There is deliberately **no `RERANK_*` knob** for it
(user decision q1): a knob would have touched the render tables, `env.example`,
37 goldens and the lock allowlist; the bake changes one compose golden.

### The judge prompt

The vendored jinja renders the model card's format:

```text
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the
Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {{ instruction }}
<Query>: {{ query }}
<Document>: {{ document }}<|im_end|>
<|im_start|>assistant
<think>

</think>
```

The default instruction, when a request supplies none, is the card's own:

```text
Given a web search query, retrieve relevant passages that answer the query
```

### Per-request `instruction`

Both `/v1/rerank` and `/v1/score` accept a **top-level `"instruction"`** field;
vLLM's `ScoringRequestMixin` folds it into `chat_template_kwargs`, and the jinja
reads it in preference to the default above.

```bash
curl -s http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Reranker-0.6B",
    "instruction": "Given an operator question about the lobes fleet, retrieve the doc passage that answers it",
    "query": "Which file lists the ports the gateway exposes?",
    "documents": [
      "The gateway port ledger is docs/gateway-fleet.md, which lists 8000 and 8001.",
      "Cats purr when they are content."
    ]
  }'
```

Before this change the field was **accepted and silently ignored** — measured
2026-08-30, identical scores and an identical 53-token prompt with and without
it, on both endpoints. It is **live now**: the same pair scored **0.9999
without** the instruction and **0.9996 with** it, at **194 → 200 prompt
tokens** (+6), on both `/v1/rerank` and `/v1/score`.

### What the scores mean — measured

Before/after on the same probe set, DGX Spark GB10, 2026-08-30
(`docs/evidence/2026-08-30-baseline-reranker-untemplated-spark.txt` →
`docs/evidence/2026-08-30-accept-reranker-template-spark.txt`):

| probe | document | before | after |
|---|---|---:|---:|
| sky | sky is blue (relevant) | 0.973 | 0.998 |
| sky | cats purr | 0.794 | 0.000 |
| sky | invoice due | 0.290 | 0.000 |
| ports-ledger | ledger (relevant) | 0.936 | 1.000 |
| ports-ledger | cats purr | 0.524 | 0.000 |
| ports-ledger | bananas | 0.510 | 0.000 |
| france (assess probe) | Paris (relevant) | 0.973 | 0.995 |
| france (assess probe) | Amazon | 0.216 | 0.000 |
| france (assess probe) | bananas | 0.876 | 0.000 |
| toolbatch inversion | NOTICE (distractor) | 0.790 | 0.000 |
| toolbatch inversion | toolbatch (relevant) | 0.988 | 1.000 |
| toolbatch inversion | cats purr | 0.741 | 0.000 |
| graded relevance | full answer (best) | 0.970 | 1.000 |
| graded relevance | terse answer (weaker) | 0.700 | 1.000 |
| graded relevance | cats purr | 0.283 | 0.000 |
| — | prompt tokens per pair | 17–30 | 88–100 |
| — | `instruction` changes the score | no | yes |
| — | latency, 1 query × 5 docs (median of 5) | 28.0 ms | 18.1 ms |

Plainly: **every distractor drops to 0.000, every relevant document scores
≥ 0.995, and the #220 NOTICE-above-toolbatch inversion is resolved** — not
merely narrowed.

**But the scores SATURATE, and that is the honest caveat.** In the graded probe
— one query with a full answer and a terse one — **both scored 1.000, and the
weaker document ranked first** (ranking `[1, 0, 2]`). So:

> **A relevance THRESHOLD is now safe. Fine ORDERING among several genuinely
> relevant documents is NOT guaranteed.**

That replaces this page's former "usable for top-k ordering, not thresholds"
framing, which described the untemplated lane. The trade is real and runs in
both directions: the untemplated lane had ordering resolution and no usable
cutoff; the templated lane has a usable cutoff and coarse resolution inside the
relevant set.

Who this matters to — the two callers in the mesh at the time of the change:

- **eidetic-cli** (`eidetic/memory/embed.py`) maps index → `relevance_score`
  and **sorts** by it. It pins no cutoff, so this is a score-*value* change, not
  an API break — but its ordering among several relevant memories is now coarse.
- **colleague's #277 retrieval lane**, which will want a cutoff: it can now use
  one.

### Document length at the 8192 ceiling

The judge prompt costs roughly **65 tokens per pair** (17–30 → 88–100 measured),
so the **effective maximum document length shrinks by about that much** under
`RERANK_MAX_MODEL_LEN=8192`. Whether vLLM truncates (`max_tokens_per_doc`) or
returns a 400 at that edge is **unexamined**.

### Telling a templated box from an untemplated one

The rerank-ordering correctness probe is **still flagged unverified on the
GB10** (issue #106) — this change does not retire that. What it does add:
`lobes assess --probes --role reranker` now reports
`prompt_tokens {total, per_pair}` in its evidence (measured **88.0/pair**
templated, vs **~17–30/pair** untemplated). That is the one external tell that a
given box is templated; the probe's PASS rule is unchanged (ordering only).

### Rollout — by-box divergence, accepted

Validated on the **DGX Spark only** (decision q3). The Jetson AGX Thor and the
Jetson AGX Orin also host `vllm-rerank` and **serve untemplated scores until
their next `lobes init --apply`**. That divergence window is accepted, not
overlooked: during it, two boxes in the same mesh answer the same rerank request
on different score scales.

### Gateway note

Approved deviation d1: the gateway relays `/v1/rerank` and `/v1/score` bodies
**semantically unmodified** — it never injects, rewrites, or defaults an
`instruction` field; the prompt is shaped inside the engine by
`--chat-template`, never by a gateway rewrite. It does **re-encode** the JSON
(whitespace), so the forwarded body is semantically identical but not
byte-identical.

## Upgrade note — a missing jinja fails LOUD

A deployment dir re-rendered with the new compose but **lacking
`qwen3_reranker.jinja`** does not degrade to untemplated scoring. Docker creates
a **directory** at the bind-mount source, vLLM's `validate_chat_template` then
fails at arg-parse, and `vllm-rerank` **crash-loops** with `/health` never going
green (the gateway advertises `reranker ready:false`).

This is loud **by design**: the alternative — a template that fails to resolve
at request time and silently falls back to `default_tokenizer_encode` — would
have reproduced exactly the uncalibrated scoring this change fixes.

Recovery, before bringing the lane up:

```bash
lobes init --apply            # or: lobes doctor --fix --apply
lobes up reranker --apply
```

On a box whose deployed `docker-compose.yml` is **hand-edited** (the #214
incident on the Spark), a re-render is not safe. The Spark took the deviation-d2
route instead on 2026-08-30: a dated backup of the live compose
(`docker-compose.yml.bak-20260830-pre-227`), the vendored jinja copied into the
deployment dir, and the **two lines applied by hand** — the `:ro` bind mount and
the `--chat-template` arg — then `lobes up reranker --apply`. The lane came up
healthy after ~115 s with `restarts=0`; see the acceptance transcript's "Deploy
record".

## API call shapes

The gateway routes `/v1/rerank` and `/v1/score` to this backend by matching
`"model": "Qwen/Qwen3-Reranker-0.6B"` — the same gateway port as chat and
embeddings.

> **The `model` field is required.** Routing is by model name, so a request
> without `model` falls through to the gateway's default (the chat primary),
> which can't score (returns a 400). Always send `model` in the request body.

### Rerank (Jina / Cohere shape — sorted best-first)

Use `/v1/rerank` when you want results ranked from most to least relevant. The
`index` in each result refers to the position in the original `documents` list.

```bash
curl -s http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Reranker-0.6B",
    "query": "What is the capital of France?",
    "documents": [
      "Paris is the capital of France.",
      "Berlin is the capital of Germany.",
      "Rome is the capital of Italy."
    ]
  }'
```

```json
{
  "results": [
    {"index": 0, "relevance_score": 0.91},
    {"index": 2, "relevance_score": 0.18},
    {"index": 1, "relevance_score": 0.07}
  ]
}
```

Results are sorted **best-first** (highest `relevance_score` first).

### Score (vLLM pairwise shape — input order)

Use `/v1/score` when you need raw scores in the original input order (e.g. to
join scores back to your document list by index without re-sorting).

```bash
curl -s http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Reranker-0.6B",
    "text_1": "What is the capital of France?",
    "text_2": [
      "Paris is the capital of France.",
      "Berlin is the capital of Germany."
    ]
  }'
```

```json
{
  "object": "list",
  "data": [
    {"index": 0, "score": 0.91},
    {"index": 1, "score": 0.07}
  ]
}
```

Results are returned in **input order** (no sorting). Use `/v1/rerank` for
sorted output with the Jina/Cohere interface.

## Health check

```bash
curl -fsS http://localhost:8000/health
curl -s http://localhost:8000/v1/models
```

## Co-residency note

This backend runs **warm alongside the 27B primary** (`Qwen3.6-27B-Text-NVFP4-MTP`)
and the embedder (`Qwen3-Embedding-0.6B`) on the single GB10. Because it uses a
classification task (no autoregressive decode), its KV cache footprint is
negligible — it does not compete with the primary for KV memory even under
concurrent reranking workloads.

The gateway routes requests by `model` field at the shared port, so reranking,
scoring, embedding, and chat calls all share one endpoint with zero client-side
port configuration.

## Composition with eidetic-cli

Typical RAG pipeline using both embed + rerank gears:

1. `POST /v1/embeddings` (Qwen3-Embedding-0.6B) → 1024-dim vectors
2. eidetic ingest — stores vectors + metadata
3. eidetic retrieve — nearest-neighbour search returns top-K candidates
4. `POST /v1/rerank` (Qwen3-Reranker-0.6B) — cross-encoder reranks top-K,
   returning the globally best passages before generation

The embedder handles recall; the reranker handles precision. Both run warm on
the same GB10 gateway port, so neither adds a new service or port to the client.

## Assessment / Benchmark

**Load-tested 2026-06-19 on the DGX Spark (GB10, 128 GB unified)** — served warm
under `--runner pooling --convert classify` + the `Qwen3ForSequenceClassification`
hf-override, `--gpu-memory-utilization 0.06`, `--max-model-len 8192`, **co-resident
with the 27B primary and the embedder** (all three simultaneously healthy):

| Metric | Result |
|---|---|
| endpoints | `/v1/rerank` (sorted) + `/v1/score` (input order) — one backend |
| rerank latency (warm, 1 query × 5 docs) | ~25 ms |
| ranking quality | relevant docs ranked first (e.g. France-capital query: Paris doc top at 0.98) |
| score endpoint | `/v1/score` returns per-pair scores ✓ |
| co-residency | 27B chat unaffected while the reranker served |
| prompt tokens per pair (untemplated, 2026-08-30) | 17–30 |
| prompt tokens per pair (templated, 2026-08-30) | 88–100 |
| rerank latency, 1 query × 5 docs, median of 5 (templated, 2026-08-30) | 18.1 ms (untemplated the same day: 28.0 ms) |

The last three rows are from the #227 calibration runs on the DGX Spark
(`docs/evidence/2026-08-30-baseline-reranker-untemplated-spark.txt` and
`docs/evidence/2026-08-30-accept-reranker-template-spark.txt`); the rows above
them remain the historical 2026-06-19 record. Latency did **not** rise despite
~3.5× the prompt tokens per pair — this lane is prefill-cheap at this size, and
both figures sit within cold-vs-warm cache noise.

Served on this vLLM build (`0.19.0+nv26.04`) with `--runner pooling --convert
classify` — the older `--task score` is rejected (`unrecognized arguments`). The
probes use plain `curl` against `/v1/rerank` and `/v1/score`.

This flag set is unchanged on the fleet's current vLLM nightly build
(`docs/vllm-nightly-migration.md` §5, t3 spike, live 2026-07-01) —
`--runner pooling --convert classify` and the `Qwen3ForSequenceClassification`
hf-override both still work identically; the numbers above were not
re-measured on nightly but the serving contract did not change.
