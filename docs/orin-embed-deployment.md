# The Orin as the mesh's embedding specialist — the `orin-embed` deployment

> **Status, 2026-10-07:** the `orin-embed` shape is deployed live on the Jetson
> AGX Orin 64GB and accepted mesh-wide
> ([acceptance](evidence/2026-10-07-accept-orin-embed.txt)). Lane
> `gemma2-embed` (EmbeddingGemma 2) is **MEASURED** on the Orin and reached by
> name from spark, spark2 and thor. **Retrieval quality beyond code is
> UNVALIDATED** (issue #296). The associate role is DORMANT/unhosted mesh-wide
> as a result; [`orin-associate-deployment.md`](orin-associate-deployment.md)
> records it and `orin-associate` is the rollback shape. Operator decisions and
> deviations: `docs/specs/2026-10-07-orin-embedding-specialist.md`.

## What the shape hosts

`lobes init --shape orin-embed --profile orin` renders
(`lobes/profiles/builtin_shapes/orin-embed.toml`):

| what | kind | notes |
|---|---|---|
| `embedder` | role (unchanged) | Qwen3-Embedding-0.6B, util 0.06, 5.24 GiB |
| `reranker` | role (unchanged) | Qwen3-Reranker-0.6B, util 0.06, 5.34 GiB in the acceptance sample |
| `gemma2-embed` | **lane** | [EmbeddingGemma 2](embeddinggemma-2.md): text, code, image, video, audio; 768-d, MRL 128/256/512/768; Sentence-Transformers sidecar; 4.69 GiB (`mem_limit` 6g) |

It hosts no associate, cortex, senses or hand. A **lane** is addressed by its
own name and is not a Colleague role (the ten-role contract is untouched; see
[`colleague-stack.md`](colleague-stack.md#specialist-lanes-not-roles)).
Budget, MEASURED: about 15.3 GiB of 61.34 GiB for the three; acceptance
recorded `free -m` available 43,531 MiB, zero kernel OOM lines, and no restarts
of the new lane.

Measured but **not carried** (all in
[`2026-10-07-spike-orin-embed-vllm-8b.txt`](evidence/2026-10-07-spike-orin-embed-vllm-8b.txt)
and [`2026-10-07-h2h-code-retrieval-orin.txt`](evidence/2026-10-07-h2h-code-retrieval-orin.txt)):

| lane | doc | status |
|---|---|---|
| `nemotron-embed` | [`nemotron-3-embed-8b.md`](nemotron-3-embed-8b.md) | declared OPT-IN candidate (deviation d1) |
| `nomic-code-embed` | [`nomic-embed-code.md`](nomic-embed-code.md) | measured, not carried |
| `qwen3vl-embed` | [`qwen3-vl-embedding-8b.md`](qwen3-vl-embedding-8b.md) | measured, not carried (util 0.52, cannot co-reside) |
| `qwen3vl-rerank` | [`qwen3-vl-reranker-8b.md`](qwen3-vl-reranker-8b.md) | EXCLUDED (c18): never served |

## Live deploy recipe

The deployment is `~/.lobes`, not this repo. **Never run `docker compose` inside
`lobes/templates/`**; use the `lobes-deploy` skill's `lobes-compose.sh`
([`operating-a-deployment.md`](operating-a-deployment.md)). The sequence the
live Orin followed:

1. **Back up first.** `cp -a ~/.lobes ~/.lobes-bak-<date>-pre-orin-embed`
   (the live run backed up 24 files including `.env`). The backup is the
   rollback; see below.
2. **Render the shape** (dry-run by default): `lobes init --shape orin-embed
   --profile orin` and read the plan, then add `--apply --force`. The render
   scaffolds `docker-compose.embed.yml` and `Dockerfile.embed-st`, appends
   `gemma2-embed` to `COMPOSE_PROFILES`, and sets
   `GEMMA2_EMBED_BASE_URL=http://embed-gemma2-embed:8000` so the gateway wires
   exactly the hosted set. `.env` is merge-only: an existing line is never
   rewritten.
3. **Declare associate unhosted.** Add `ASSOCIATE_FEASIBLE=false` by hand
   (an explicit `false` is what makes the gateway answer honestly and lets the
   mesh route the role; an absent key hard-404s). The re-render does not do it.
4. **The compose chain** is the container's own `-f` chain plus the overlay:
   `docker-compose.yml`, `docker-compose.gpu.yml`, `docker-compose.shape.yml`,
   `docker-compose.override.yml`, `docker-compose.embed.yml`. Take it from
   `lobes fleet files` or `lobes-compose.sh` rather than typing it.
5. **Build and up single services**, always `--no-deps` (issue #222: `up`
   walks `depends_on`):

   ```bash
   lobes-compose.sh --apply build embed-gemma2-embed
   lobes-compose.sh --apply --profile gemma2-embed up -d --no-deps embed-gemma2-embed
   lobes-compose.sh --apply up -d --build --no-deps gateway
   ```

   The 0.6B embedder and reranker containers are never recreated. Equivalent
   lane verbs: `lobes up gemma2-embed --apply`, `lobes status` (lists defined
   lanes), `lobes assess gemma2-embed` (probes one lane THROUGH this box's
   gateway by lane name, so it works on any mesh member; `--endpoint
   http://127.0.0.1:<port>` probes a lane server directly).
6. **Dev wheels.** While the sidecar and gateway need an unreleased lobes-cli,
   set `GATEWAY_PIP_EXTRA_INDEX_URL` in `.env` to the TestPyPI index; the
   Dockerfile fetches the `.devN` wheel `--no-deps` from that index only
   (`LOBES_DEV_INDEX_URL`) and resolves the rest from PyPI. Leave it empty on
   release pins. A just-published dev wheel can take 1 to 2 minutes to
   propagate; wait and retry. Re-image **every** mesh member's gateway
   (`up -d --no-deps gateway`) so each advertises the lane identity; the live
   run used lobes-cli `0.84.0.dev619` on all four.
7. **Verify** with `GET /capabilities` (a `gemma2-embed` key with
   `lane: true`), a by-name `POST /v1/embeddings` from a non-hosting member and
   a per-modality probe, as the acceptance transcript does.

## Two Jetson traps (both MEASURED 2026-10-07)

- **CUDA "free" excludes the page cache.** After heavy file I/O (a model
  download or a benchmark pass) vLLM saw 18.1 GiB "free" with about 30 GiB
  "available" and refused to boot at util 0.35. Evict the Hugging Face cache
  pages without root before booting:

  ```python
  import os, pathlib
  for p in pathlib.Path.home().joinpath(".cache/huggingface").rglob("*"):
      if p.is_file():
          fd = os.open(p, os.O_RDONLY)
          os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
          os.close(fd)
  ```

  This cut buff/cache from 18.5 to 3.8 GiB and both 8B engines then booted.
- **GPU memory counts against the container's cgroup `mem_limit`.** Unified
  memory means a vLLM engine's GPU allocations are charged to the container.
  Both Qwen3-VL engines were OOM-killed at `--memory 30g` mid weight-load. Size
  every lane's `mem_limit` for weights + KV + encoder profile + graphs, not
  host RSS alone. The Orin also has zero swap, so cap any side work with
  `--memory` too.

## Opting in Nemotron

`nemotron-embed` is not rendered by the shape (deviation d1). To run it beside
`gemma2-embed`, add `nemotron-embed` to `COMPOSE_PROFILES` and set the MEASURED
knobs in `.env`:

```bash
NEMOTRON_EMBED_GPU_MEM_UTIL=0.35
NEMOTRON_EMBED_MAX_MODEL_LEN=8192
NEMOTRON_EMBED_MEM_LIMIT=26g
NEMOTRON_EMBED_BASE_URL=http://embed-nemotron-embed:8000
```

Then `lobes-compose.sh --apply --profile nemotron-embed up -d --no-deps
embed-nemotron-embed` and re-image the gateway. Measured: 20.56 GiB resident;
with it added too, 18.7 GiB stayed available through a full head-to-head. Why it
is opt-in, and what it did and did not show, is in
[`nemotron-3-embed-8b.md`](nemotron-3-embed-8b.md). Its 4096-d vectors are
another space again.

## Rollback

From [`2026-10-07-rollback-orin-embed-to-orin-associate.txt`](evidence/2026-10-07-rollback-orin-embed-to-orin-associate.txt),
rehearsed on a copy of the live deployment (the live box was not rolled back):
**a re-render alone is NOT a complete rollback; restoring the backup is.**
`lobes init --shape orin-associate --profile orin --apply --force` restores the
associate boot edges and `COMPOSE_PROFILES`, but `.env` is merge-only and init
never deletes files, so it leaves behind: `ASSOCIATE_FEASIBLE=false` (keeps
associate off), the `GEMMA2_EMBED_*` / `NEMOTRON_*` knobs (the gateway would
still advertise a lane no profile starts), and `docker-compose.embed.yml` +
`Dockerfile.embed-st` in the chain. `MODEL_GEAR_VERSION` is re-rendered to the
CLI's own version; pin it back. The complete rollback, in order:

1. Stop and remove the lane: `stop embed-gemma2-embed`, then `rm -f
   embed-gemma2-embed` (with the embed overlay still in the chain).
2. Restore the backup over the deployment (`cp -p <backup>/* <backup>/.env
   ~/.lobes/`) and remove `docker-compose.embed.yml` and `Dockerfile.embed-st`
   (or delete `ASSOCIATE_FEASIBLE`, restore `ASSOCIATE_BASE_URL` and comment
   out the lane keys by hand).
3. Re-image the gateway: `up -d --build --no-deps gateway` over the associate
   chain (no embed overlay).

Associate was already `Exited (1)` before the change (cause not diagnosed), so
serving it again was never part of the claim.

## Consumers

Nothing consumes the lane until its owner says so, and **no consumer is
switched implicitly**. The two embedding consumers in the workspace read their
model from the environment and both default to `Qwen/Qwen3-Embedding-0.6B`:

| consumer | variable | default |
|---|---|---|
| eidetic-cli | `EIDETIC_EMBED_MODEL` | `Qwen/Qwen3-Embedding-0.6B` |
| coherence-cli | `COHERENCE_EMBED_MODEL` | `Qwen/Qwen3-Embedding-0.6B` |

Opting a consumer in is a per-consumer, explicit act: set the variable to the
lane id, or to the raw model id, which routes to the same lane:

```bash
export EIDETIC_EMBED_MODEL=gemma2-embed          # or: google/embeddinggemma-2
export COHERENCE_EMBED_MODEL=gemma2-embed
```

**Switching means re-embedding.** EmbeddingGemma 2's 768-d vectors are a
different space from the 0.6B's 1024-d; a corpus indexed with one can be queried
only with the same one, and a mismatch silently returns meaningless scores
(MRL truncation does not bridge it). Re-embed the whole store, or keep the old
model for the old index. There is **no cross-lane fallback**: a stopped or
unwired lane answers 503 or 404, never a vector from another model, so a
consumer cannot drift by accident. Whether `gemma2-embed` retrieves *better*
than the 0.6B for a given corpus is UNVALIDATED outside code (#296).
