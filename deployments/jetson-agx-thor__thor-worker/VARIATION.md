# `jetson-agx-thor__thor-worker` — Thor serving `worker` on Qwen3.6-35B-A3B

## What this variation is

A **Jetson AGX Thor** (sm_110, 122.8 GiB unified, L4T R38.2.2, MAXN) running
the **`thor-worker`** shape over the **`thor`** card profile, captured from the
live box on 2026-09-11 at lobes 0.74.6. It was **re-captured the same day at
0.75.1**, after the reranker judge-prompt template (#227) was rolled out to it.
**Re-captured again at 0.76.0** (mesh-brain-join docs task), this time
against the repo's current templates rather than the live box — no capture
CLI verb exists (see "Notes" below), and this repo has no reachable Thor —
so `docker-compose.yml`/`docker-compose.shape.yml` and the `[env]` table now
carry the mesh join's `LOBES_MESH_*` compose passthrough and the issue #222
`depends_on` cleanup. `docker-compose.override.yml` (hand-authored) and every
other file are untouched. The `[env]` table's `PRIMARY_*`/`MULTIMODAL_*`
"documentation only" values (both roles are `FEASIBLE=false` on this shape)
now reflect a fresh render of the current `thor` card profile's
machine-as-brain defaults rather than the live box's deviation-d1-era
leftover local-cortex tuning the previous captures preserved — this capture
no longer claims those specific numbers describe the live box, only what a
fresh render of today's templates produces for a dropped role's
documentation fields. Nothing about the **measured** state below changed.
**Re-captured again** (t14, env peer family retirement): `docker-compose.yml`
lost its `*_PEER_ORIGIN(S)`/`*_PEER_PROXY`/`*_PEER_API_KEY(S)` gateway
passthrough lines — `lobes.gateway._config.build_config` no longer reads any
of them, so a deployed value under one of those names is now silently inert
regardless of whether the compose file still carries the line. Same "no
reachable Thor, no capture CLI verb" method as the previous re-capture: the
packaged template's `docker-compose.yml` was copied in verbatim and its
digest recomputed. `docker-compose.shape.yml`/`docker-compose.override.yml`
and every other file are untouched, and the `[env]` table is unaffected (it
never carried a `*_PEER_*` entry — those are operator-declared, never
profile-rendered). The paragraph below describing `PRIMARY_PEER_ORIGIN` +
`PRIMARY_PEER_PROXY` forwarding `model=cortex` to a Spark peer is historical
narrative of how this box reached cortex before the mesh-brain join
replaced that mechanism (see CLAUDE.md's "Live as of 2026-09-11" paragraph)
— it is not re-asserted as current operator guidance by this capture.

It hosts **`worker`** — `nvidia/Qwen3.6-35B-A3B-NVFP4` at the full native
262144 window, `gpu_mem_util=0.45`, fp8 KV cache, `max_num_seqs=1`, and DFlash
k=12 speculation against the `z-lab/Qwen3.6-35B-A3B-DFlash` drafter — plus the
`embedder` and `reranker` pooling gears. The reranker serves with
`--chat-template` pointing at the vendored `qwen3_reranker.jinja`, which is
committed here.

It does **not** host `cortex`: `PRIMARY_FEASIBLE=false`, with
`PRIMARY_PEER_ORIGIN` + `PRIMARY_PEER_PROXY` forwarding `model=cortex` to a
DGX Spark peer. `hand` is declared infeasible on this card (a measured sm_110
inference defect, not a budget choice). `senses` is also infeasible, with **no**
peer: it used to be proxied to a Jetson AGX Orin, but the Orin does not host
it, so `model=senses` now 404s `role_infeasible`.

An adopter needs, before running it:

* the same card — the lock refuses a variation mismatch by design;
* a reachable Spark-shaped peer for `cortex`, or an edit to the peer origin;
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
> `docs/evidence/2026-09-11-accept-thor-reranker-template-senses-unproxy.txt`
> covers the state captured here. The templated reranker went from 0.21–0.87
> distractor scores to 0.000 at three decimals (raw values are about 1e-5 and
> strictly positive), and the `instruction` field is now honoured
> (43.2 → 47.3 ms median). `senses` now 404s honestly, and the
> cortex/worker/embedder proxy paths show no regression.
> `docs/evidence/2026-09-10-accept-thor-worker-flip.txt` covers the flip
> itself: worker serving locally, `cortex` answering by proxy with
> `X-Lobes-Proxied-By`, single-stream decode through the gateway, the
> structured tool-call probe, image intake with negative controls, and an
> end-to-end agentic run driven by Qwen Code.
> `docs/evidence/2026-09-11-accept-worker-proxy-spark-thor.txt` covers a DGX
> Spark reaching this box's `worker` cross-box.
>
> Not covered: long-context retrieval at 262144 (115K is the deepest
> measured), video intake, and serve-after-restore. This capture has never
> been restored onto a fresh box and then served.

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
two knob slots described above, plus the two #227 reranker lines (the jinja
bind mount and `--chat-template`), added by hand rather than by re-rendering.
It is captured verbatim rather than regenerated, which is the entire point of
a lock: the bytes that produced the measurements, not the bytes a current
render would produce.

**`.env` is NOT captured**, not even by digest. `lobes init --from-lock`
refuses any `.env`-family name in `[files]` outright (`_check_restorable_name`
in `lobes/cli/_commands/init.py`: "the .env family is never committed"). The
catalog validator also requires every `[files]` entry to exist in this
directory, and a deployed `.env` can never be committed. Listing its digest
would therefore make the entry unrestorable. The restorable settings live in
`[env]`. The
`.env`-digest drift check (deviation d4) therefore applies only to a lock kept
beside a live deployment, never to a catalog entry.

**Re-capture after any change.** There is no capture verb; the writer is a
library (`lobes/runtime/_lock.py`'s `capture_lock`). Re-run it after a
`lobes switch` or a hand edit, or the drift check will fire.
