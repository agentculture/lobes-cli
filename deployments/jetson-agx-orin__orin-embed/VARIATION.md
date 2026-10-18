# `jetson-agx-orin__orin-embed` — the Orin as the mesh's embedding specialist

## What this variation is

A **Jetson AGX Orin 64GB** (sm_87, 61.34 GiB unified, ZERO swap, L4T R39)
running the **`orin-embed`** shape over the **`orin`** card profile, captured
from the live box on 2026-10-07 at lobes `0.84.0.dev619` (draft PR #295,
issue #291). It hosts:

- the unchanged 0.6B `embedder` and `reranker` roles;
- the **`gemma2-embed`** specialist lane. This is `google/embeddinggemma-2`
  (text, code, image, video and audio in one 768-d space) served by a
  Sentence-Transformers sidecar (`Dockerfile.embed-st`), because the Orin's
  vLLM digest refuses the checkpoint.

`associate` is dropped (`ASSOCIATE_FEASIBLE=false`). `nemotron-embed` is
declared but not hosted: it is opt-in, per deviation d1.

The files here are the live deployment's, verbatim.

- `docker-compose.override.yml` is hand-authored (the mesh passthrough).
- The `docker-compose.yml` / `docker-compose.gpu.yml` /
  `docker-compose.embed.yml` / `docker-compose.shape.yml` chain is what
  `lobes init --shape orin-embed --profile orin` rendered.
- The live box's gateway chain is `docker-compose.yml`,
  `docker-compose.gpu.yml`, `docker-compose.embed.yml`,
  `docker-compose.shape.yml` and `docker-compose.override.yml`. It runs
  without the audio overlay, which this board never ran.

**Re-captured from the LIVE box on 2026-10-18 at 0.89.0** (#310 t16, after the
fleet moved to vLLM 0.31.0 `v0.31.0-ubuntu2404`). The files are what the Orin
holds now (fetched read-only over ssh; `.env` was read for the allowlist only
and is not committed). The box now also runs a `decider-general` sidecar
(`COMPOSE_PROFILES=decider,gemma2-embed`), so `Dockerfile.decider-vllm` joins
the captured files and the `[env]` table gained the `DECIDER_DOMAIN*` keys,
`LOBES_BOOT_ORDER` and `LOBES_THERMAL_ZONES`; `GEMMA2_EMBED_MEM_LIMIT` moved
from 6g to 8g. The engine pin is a `.env` key the lock's allowlist does not
record, so it is not in this directory.

## Measured result

Measured: `docs/evidence/2026-10-17-accept-vllm-031-orin.txt` covers the
state captured here: embed, rerank, gemma2-embed and decider-general all
serving on vLLM 0.31.0 (sm_87), with probes passing, the embedder measured at
roughly 13% lower throughput than the nightly (recorded there as a named
cost), and the decider re-measured at 0.9928 agreement with bf16.

Earlier measurements of the pre-0.31.0 engine:
`docs/evidence/2026-10-07-accept-orin-embed.txt` (mesh-wide
acceptance), with the lane footprint in
`docs/evidence/2026-10-07-spike-embeddinggemma2-sidecar-orin.txt` and the
code head-to-head in `docs/evidence/2026-10-07-h2h-code-retrieval-orin.txt`.

## Notes

- **Two Jetson traps this deployment depends on:**
  - GPU memory counts against a container's `mem_limit`.
  - CUDA "free" excludes page cache, so a lane can refuse to boot after
    heavy I/O until the Hugging Face cache pages are evicted. See
    `docs/orin-embed-deployment.md`.
- **Rollback:** restore the pre-change backup. A re-render of
  `orin-associate` alone is incomplete
  (`docs/evidence/2026-10-07-rollback-orin-embed-to-orin-associate.txt`).
- **Quality:** only code retrieval was quality-tested. Image, video and audio
  were checked with synthetic ordering probes. The real evaluation is issue
  #296.
- **No capture verb exists.** This lock was written by calling
  `lobes.runtime._lock.capture_lock` / `write_lock` directly, over the live
  `.env` (read only, never committed).
