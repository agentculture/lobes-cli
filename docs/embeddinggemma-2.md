# EmbeddingGemma 2 — the `gemma2-embed` specialist lane (768-d, five modalities)

> One entry in lobes's **supported catalog** (`lobes overview --list`). For the
> catalog-vs-warm distinction see
> [`gateway-fleet.md`](gateway-fleet.md#supported-catalog-vs-warm-backends).
>
> **Status: `load-tested` on the Jetson AGX Orin 64GB only (2026-10-07).**
> MEASURED on the Orin: the sidecar load, per-modality encode paths, the
> mesh-wide by-name acceptance and the code head-to-head. **Retrieval quality
> beyond code is UNVALIDATED** (issue #296, see "What is NOT validated").
> Transcripts: [spike](evidence/2026-10-07-spike-embeddinggemma2-sidecar-orin.txt),
> [acceptance](evidence/2026-10-07-accept-orin-embed.txt),
> [code head-to-head](evidence/2026-10-07-h2h-code-retrieval-orin.txt).

## What it is

`google/embeddinggemma-2` is the **only standard specialist lane** of the
`orin-embed` shape (deviation d1; deployment recipe in
[`orin-embed-deployment.md`](orin-embed-deployment.md)). It is a **lane**, not a
Colleague role: it is addressed by its own name `gemma2-embed` and the ten-role
contract is untouched (see
[`colleague-stack.md`](colleague-stack.md#specialist-lanes-not-roles)).

- Apache-2.0, ~740M parameters (PUBLISHED-ELSEWHERE: Hugging Face card, read
  2026-10-07; not measured here beyond the 744.4M parameter count below).
- **One shared 768-d space** across text (including code), image, video and
  audio.
- **Matryoshka (MRL)** dimensions 128 / 256 / 512 / 768; the sidecar truncates
  and re-normalizes to unit length.
- 8192-token window (`native_max_model_len=8192`).
- Catalog `engine`: `sentence-transformers` (not vLLM).
- **Served name == catalog id:** `google/embeddinggemma-2`; the lane name
  `gemma2-embed` and the raw id both route to it.
- `role_hint="candidate"`: it can never hijack the 0.6B `embedder` role.

## Why a sidecar, not vLLM

MEASURED 2026-10-07: the Orin's pinned engine image
(`vllm/vllm-openai@sha256:7c5a10e9...`, vllm `0.23.1rc1.dev672+g93d8f834d`,
transformers 5.12.1) **refuses the checkpoint** at config validation: model type
`embedding_gemma2`, "Transformers does not recognize this architecture"
([spike](evidence/2026-10-07-spike-embeddinggemma2-sidecar-orin.txt)). A
Sentence-Transformers image built **on that same engine base** serves it:

| package | version in the image |
|---|---|
| transformers | 5.19.0 (a release; the card's config names 5.18.0.dev0) |
| sentence-transformers | 6.1.0 |
| torch | 2.11.0+cu130 (not the card's 2.14; the base image's torch works) |
| Pillow | 12.2.0 |

`lobes/templates/fleet/Dockerfile.embed-st` pins exactly that base digest
(`ST_BASE_IMAGE`) and `lobes-cli[embed-sidecar]==${MODEL_GEAR_VERSION}`. The
base digest is deliberately **not** the fleet's `VLLM_NIGHTLY_IMAGE`; it is the
one the spike measured. bf16 only: `EMBED_DTYPE=float16` is refused at load
because the checkpoint returns NaN or silently degraded vectors in float16.

## Serving

The service is `embed-gemma2-embed` in `docker-compose.embed.yml`, behind the
compose profile `gemma2-embed`, publishing no host port. The gateway reaches it
at `http://embed-gemma2-embed:8000` once `GEMMA2_EMBED_BASE_URL` is set (the
shape render does this). Sidecar environment:

| variable | default | meaning |
|---|---|---|
| `EMBED_MODEL_ID` | `google/embeddinggemma-2` | checkpoint id or a local path (fine-tunes) |
| `EMBED_SERVED_NAME` | `EMBED_MODEL_ID` | identity reported in `model` and `/health` |
| `EMBED_MODALITIES` | `text` (compose passes `text,image,video,audio`) | selective load via the card's `config_kwargs`; `text` is mandatory |
| `EMBED_DTYPE` | `bf16` | `bf16` or `fp32`; `float16` is refused |

`GET /health` answers 503 `{"status":"loading"}` until the model is loaded, then
200 with `model`, `dtype` and `modalities_loaded`.

## Requests

Everything below goes through the gateway (`POST /v1/embeddings`) with
`"model": "gemma2-embed"` (or `"google/embeddinggemma-2"`).

Text:

```bash
curl -s localhost:8000/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model": "gemma2-embed", "input": ["hello world", "second text"]}'
```

A named prompt (the checkpoint's own prompt table; there is no default prompt,
the card's `default_prompt_name` is null). An unknown name is a 400
`unknown_prompt_name` listing the known ones. The response echoes `prompt_name`
and `prompt`:

```bash
curl -s localhost:8000/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model": "gemma2-embed", "input": "def merge(a, b): ...", "prompt_name": "CodeRetrieval"}'
```

MRL dimensions (128, 256, 512 or 768; anything else is a 400
`unsupported_dimensions`):

```bash
curl -s localhost:8000/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model": "gemma2-embed", "input": "hello", "dimensions": 128}'
```

Image, video or audio: use a `messages` body (exactly one of `input` or
`messages` per request). All parts across all messages form **one** input and
yield **one** embedding. Media must be `data:` URLs; an `http(s)://` URL is a
400 `remote_url_unsupported` because the sidecar never fetches:

```json
{
  "model": "gemma2-embed",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,<B64>"}}
  ]}]
}
```

The same shape takes `{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,<B64>"}}`,
`{"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,<B64>"}}` or
`{"type": "input_audio", "input_audio": {"data": "<B64>", "format": "wav"}}`, and
may mix a `text` part with one media part (text+image). Rules:

- One part per modality per input; a second image in one input is refused, not
  reordered.
- A modality not loaded (`EMBED_MODALITIES`) is a 400 `modality_not_loaded`,
  never dropped and never captioned.
- A degenerate vector (NaN, infinity, all zeros, before or after truncation) is
  a 500 `degenerate_embedding`, never a 200.
- `encoding_format` is `float` (default) or `base64` (little-endian float32).
- `usage` token counts are reported as 0; the sidecar does not count tokens.

## Measured numbers

All MEASURED on the Orin, 2026-10-07.

Sidecar load, one process per load
([spike](evidence/2026-10-07-spike-embeddinggemma2-sidecar-orin.txt)):

| load | params | load time | CUDA peak | process RSS max | text p50 |
|---|---|---|---|---|---|
| full (all modalities) | 744.4M | 18.8 s | 1.48 GiB | 4.03 GiB | 129.1 ms |
| text only | 271.0M | 4.2 s | 0.514 GiB | 2.25 GiB | 117.4 ms |

Text, prompted text, image, audio, text+image and video each returned a finite
768-d vector with norm about 1.0 (0.9986 to 1.0029 in bf16).

As served ([head-to-head](evidence/2026-10-07-h2h-code-retrieval-orin.txt),
[acceptance](evidence/2026-10-07-accept-orin-embed.txt)): **4.69 GiB** resident
with all four modalities (`mem_limit` 6g; 1.99 GiB at rest in the acceptance
sample), restarts 0.

Code retrieval, nDCG@10 (corpus of 1,724 function/class chunks; 48 `nl_to_code`
and 24 `issue_to_source` labelled queries):

| family | EmbeddingGemma 2 | Qwen3-Embedding-0.6B |
|---|---|---|
| `nl_to_code` (n=48) | 0.9424 | 0.9472 |
| `issue_to_source` (n=24) | 0.5229 | 0.5191 |

Decision c26 applied as committed (beats EG2 by 5 points or more on **both**
families): no 8B candidate cleared the bar, so **EmbeddingGemma 2 keeps the
code slot**. The `nl_to_code` family is saturated (every model scored 0.88 to
0.95), so it cannot separate candidates; only `issue_to_source` discriminates.

Mesh acceptance (via the Spark, which hosts no embed lane): by-name requests
for `gemma2-embed` and the raw id answered 200 from spark, spark2 and thor with
`X-Lobes-Mesh-Member: orin`; per-modality negative controls ranked the correct
item first of three for text to image, text to audio, text to video and text to
code (ALL_PASS); `dimensions=128` returned a 128-d unit vector;
`dimensions=100` returned 400; a stopped lane answered 503
`backend_unavailable`, never another model's vector.

## What is NOT validated (issue #296)

Nothing beyond code retrieval was quality-tested, and even that is a 72-query
set with no confidence interval. In particular UNVALIDATED:

- retrieval quality on prose, documentation, articles, stories and math;
- real image, video and audio retrieval: the acceptance probes use **synthetic**
  inputs (solid colours, a 440 Hz tone against noise and near-silence, ffmpeg
  colour clips) and prove each modality's path and ordering, not quality;
- MRL quality: 128/256/512 are shown to return unit vectors, not to retain
  retrieval quality;
- cross-model agreement and a redone, harder code test (the `nl_to_code` set is
  saturated).

Also open: `usage` is always 0, concurrent-load behaviour of the sidecar is not
measured, and the card's torch 2.14 was not tried.

## The one rule: its vectors are a different space

EmbeddingGemma 2 vectors are **not comparable** with the 0.6B embedder's
(1024-d) or any other model's. Switching a consumer means re-embedding its whole
corpus; see the Consumers section of
[`orin-embed-deployment.md`](orin-embed-deployment.md#consumers). The gateway
gives a lane **no cross-lane fallback**: an unwired or stopped lane is a
404/503, never an answer from another vector space.

## Adding a fine-tune lane

A fine-tune of this checkpoint is served by another sidecar instance and
addressed by its own lane name (`lobes/embed_lanes.py`,
`EMBED_FINETUNE_LANES`):

1. Put the checkpoint directory on the box and declare it in `.env`, absolute
   path required, comma-separated entries: `EMBED_FINETUNE_LANES=my-tune=/abs/path/to/checkpoint`.
2. The lane name must be lowercase alphanumerics joined by `-` and must not
   collide with a role, tier or backend alias, an existing lane, or the
   `{role}-{member}` member-lane pattern. It inherits `gemma2-embed`'s engine,
   modalities and dimensions.
3. Its served identity is `local:<name>` (an HF repo id always contains `/`, so
   it can never equal a catalog id). The sidecar instance points
   `EMBED_MODEL_ID` at the checkpoint path and `EMBED_SERVED_NAME` at
   `local:<name>`.
4. Wire it like any lane: `MY_TUNE_BASE_URL`, `_FEASIBLE`, `_MAX_ACTIVE`,
   `_TESTED_ON`, `_MAX_MODEL_LEN` (the lane name upper-cased, `-` to `_`).
   The gateway builds the fine-tune's backend, its `model=my-tune` /
   `model=local:my-tune` aliases and its `/capabilities` key from
   `EMBED_FINETUNE_LANES` in ITS OWN environment (unit-tested,
   `tests/test_embed_finetune.py::test_gateway_wires_a_declared_finetune_lane`).
5. **What you add by hand.** The overlay ships no service for a fine-tune, so
   add one to `docker-compose.override.yml`: a copy of `embed-gemma2-embed`
   named `embed-my-tune` with `EMBED_MODEL_ID=/abs/path/to/checkpoint` (mount
   it) and `EMBED_SERVED_NAME=local:my-tune`. The gateway does not read `.env`,
   so also pass `EMBED_FINETUNE_LANES` and the `MY_TUNE_*` keys to the
   `gateway` service's `environment:` there. `lobes up/status/assess` know
   only the built-in lanes, and a fine-tune lane is not forwarded across the
   mesh: address it on the box that serves it.

A fine-tune is its own vector space; do not mix its vectors with the base's.
No fine-tune lane has been served on a live box (UNVALIDATED, #108).

## See also

- [`orin-embed-deployment.md`](orin-embed-deployment.md) — the shape and the live recipe
- [`nemotron-3-embed-8b.md`](nemotron-3-embed-8b.md), [`nomic-embed-code.md`](nomic-embed-code.md),
  [`qwen3-vl-embedding-8b.md`](qwen3-vl-embedding-8b.md), [`qwen3-vl-reranker-8b.md`](qwen3-vl-reranker-8b.md)
- [`qwen3-embedding-0.6b.md`](qwen3-embedding-0.6b.md) — the `embedder` role's checkpoint
