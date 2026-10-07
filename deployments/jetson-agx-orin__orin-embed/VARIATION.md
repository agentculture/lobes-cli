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

## Measured result

Measured: `docs/evidence/2026-10-07-accept-orin-embed.txt` (mesh-wide
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
