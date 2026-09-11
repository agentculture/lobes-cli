# `jetson-agx-thor__thor-worker` — Thor serving `worker` on Qwen3.6-35B-A3B

## What this variation is

A **Jetson AGX Thor** (sm_110, 122.8 GiB unified, L4T R38.2.2, MAXN) running
the **`thor-worker`** shape over the **`thor`** card profile, captured from the
live box on 2026-09-11 at lobes 0.74.6.

It hosts **`worker`** — `nvidia/Qwen3.6-35B-A3B-NVFP4` at the full native
262144 window, `gpu_mem_util=0.45`, fp8 KV cache, `max_num_seqs=1`, and DFlash
k=12 speculation against the `z-lab/Qwen3.6-35B-A3B-DFlash` drafter — plus the
`embedder` and `reranker` pooling gears.

It does **not** host `cortex`: `PRIMARY_FEASIBLE=false`, with
`PRIMARY_PEER_ORIGIN` + `PRIMARY_PEER_PROXY` forwarding `model=cortex` to a
DGX Spark peer. `hand` is declared infeasible on this card (a measured sm_110
inference defect, not a budget choice) and `senses` is proxied to a Jetson AGX
Orin peer.

An adopter needs, before running it:

* the same card — the lock refuses a variation mismatch by design;
* a reachable Spark-shaped peer for `cortex` and an Orin-shaped peer for
  `senses`, or edits to the peer origins;
* `VLLM_GDN_DECODE_KERNEL=triton` in the environment — without it, speculative
  decoding does not run on sm_110 at all;
* the pinned nightly `vllm/vllm-openai@sha256:8bd082c2…`, which the compose
  files carry verbatim.

Peer origins and credentials are **NOT** in this capture. The `[env]` table is
an allowlist of keys `lobes/profiles/render.py` renders, so operator-typed
wiring (`*_PEER_ORIGIN`, `*_PEER_API_KEY`, `*_BASE_URL`) and every credential
are absent by construction. Declare them yourself after restoring.

## Measured result

> Measured live on 2026-09-11:
> `docs/evidence/2026-09-10-accept-thor-worker-flip.txt`. Covers the flip
> itself (worker serving locally, `cortex` answering by proxy with
> `X-Lobes-Proxied-By`), single-stream decode through the gateway, the
> structured tool-call probe, image intake with negative controls, and an
> end-to-end agentic run driven by Qwen Code. Not covered: any peer reaching
> this box's `worker` cross-box (no peer declares `WORKER_PEER_ORIGIN` for it),
> long-context retrieval at 262144, video intake, and serve-after-restore —
> this capture has never been restored onto a fresh box and then served.

Companion transcripts, same box and dates:
`2026-09-10-accept-nvidia-35b-a3b-thor.txt` (load, MoE/attention backend
selection, budget), `2026-09-10-sweep-mtp-depth-nvidia-35b-a3b-thor.txt` (the
speculation sweep), `2026-09-10-baseline-thor-cortex-pre-244.txt` (the
incumbent this replaced), `2026-09-10-thor-pressure-floor-244.txt`, and
`2026-09-10-thor-backup-and-rollback-dryrun-244.txt`.

## Notes

**Why this capture exists.** The recipe is reproducible from this directory
**without re-rendering it** — `lobes init --from-lock deployments/jetson-agx-thor__thor-worker`
materialises the committed files verbatim and appends only missing `.env` keys.
That matters here because two of the settings that make the measured numbers
reproducible are **not** obtainable from a plain render of an older tree:
`WORKER_KV_CACHE_DTYPE` and `WORKER_MAX_NUM_SEQS` only became renderable in
0.74.0, and the deployed `docker-compose.yml` in `files/` was hand-patched to
add their slots before the shape could express them.

**The compose file here is hand-edited and that is deliberate.** It carries the
two knob slots described above. It is captured verbatim rather than
regenerated, which is the entire point of a lock: the bytes that produced the
measurements, not the bytes a current render would produce.

**`.env` is captured by DIGEST only**, not by content — its digest is in
`[files]` so `lobes doctor` reports `lock_drift` when the deployed file moves.
The restorable settings live in `[env]`.

**Re-capture after any change.** There is no capture verb; the writer is a
library (`lobes/runtime/_lock.py`'s `capture_lock`). Re-run it after a
`lobes switch` or a hand edit, or the drift check will fire.
