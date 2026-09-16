# ComfyUI: the `innereye` render tenant

> One entry in lobes's role contract (`docs/colleague-stack.md`) — the
> **eleventh** first-class, Colleague-facing role, and the first that is not
> a vLLM lane at all. innereye (`agentculture/innereye`, issue #82/#268) is
> today's only client.
>
> **Status: DECLARED — memory budget declared, mechanism wired, acceptance
> pending. Per the #108 honesty rule, nothing in this doc, `lobes
> capabilities`, or `GET /capabilities` may claim `innereye` VALIDATED until
> the task-t13 live acceptance transcript lands under `docs/evidence/`.**
> Two evidence transcripts exist today and neither is that one:
> `docs/evidence/2026-09-16-baseline-comfyui-venv-spark.txt` (the bare-venv
> baseline: 31.4 GiB staged, 42.05s, 85/121 GiB host memory with ComfyUI as
> the ONLY tenant on the box) and
> `docs/evidence/2026-09-16-t2-comfyui-container-spark.txt` (the
> containerized comparison: identical staging, 41.84s, and comfy-aimdo
> reading HOST totals from inside the container). Both measure ComfyUI —
> bare and containerized — in isolation. **Neither involves the gateway**:
> no run in either transcript goes through `/v1/render`, proves the 401
> negative control, or exercises the job-scoped facade. Until t13 lands,
> treat every claim below about the lobes-side surface (the facade, the
> readiness probe, the co-residency veto, the mesh non-advert) as DECLARED
> and UNVALIDATED, and treat the two existing transcripts as evidence for
> the ComfyUI workload itself, not for lobes hosting it.

## What it is

ComfyUI (`comfyanonymous/ComfyUI`, pinned v0.33.2), running as a locally
built compose service (`comfyui`) with NVIDIA's DynamicVRAM/aimdo allocator
compiled in, fronting the fleet's diffusion/image-generation workload. It
replaces a hand-started, unsupervised foreground venv process — no systemd
unit, no cron entry, no restart policy, entirely outside lobes' accounting —
with a compose-managed tenant that inherits the same lifecycle every other
lane already has: restart policy, `env_file` chaining, the GPU
`deploy.resources.reservations.devices` block, `mg-logwrap` logging, and
`lobes up <role>` / `lobes up <role> --down`.

**Role:** `innereye` — the eleventh first-class Colleague role
(`lobes/roles.py`; `docs/colleague-stack.md`), the operator decision the
`agentculture/innereye` fork issue (#268) asked lobes to make. It carries
**no `SupportedModel` catalog entry and no `role_hint`** — ComfyUI is a
render server, not a switchable checkpoint, so there is nothing in
`lobes/catalog.py` for it to cite — exactly the escape `stt`/`tts` already
use. Unlike `stt`/`tts`, `innereye` is **opt-in like `muse`/`worker`/
`associate`**: an unwired `innereye` defaults to **infeasible**, not the
audio overlay's sleeping-lobe default, so `model=innereye` on a box that
doesn't host it 404s `role_infeasible` rather than reading as a
merely-not-yet-warm generate lane.

**Backend name:** `innereye` (its own backend/service name, like the audio
sidecars). **Runtime:** `comfyui`. **Endpoint:** `POST /v1/render` — a
single facade path (`ROLE_PATH["innereye"] = "/v1/render"`), under which the
gateway owns the fan-out to ComfyUI's own submit/poll/artifact-fetch
endpoints, mirroring how `/v1/audio/*` fans out to the realtime bridge
without model-field routing. `ROLE_PATH` stays one string per role — no
other role's row changed to make room for this one.

**Responsibilities:** `image_generation`. **Forbidden:** `final_decision`,
`repo_action`, `security_decision` — innereye renders on request; it never
decides, and it never touches the repo. It is deliberately **absent** from
`ROLE_ROLE_HINT`, so `GATEWAY_FRONTED_ROLES` excludes it (again, exactly as
`stt`/`tts` do) — `innereye` never appears in `/v1/models` or the
OpenAI-shaped `model=` alias space; it is reached only by its own path.

This is the ELEVENTH role added to `docs/colleague-stack.md`'s table. The
irreversibility box that doc and `lobes/roles.py:95-118` both carry —
"reach for a new role only when a catalog change, a profile/shape change,
or a responsibilities token cannot express the need" — was not re-litigated
here; it was the operator's own decision (recorded as claim `c15` in
`docs/specs/2026-09-16-innereye-lobes-hosts-comfyui.md`), made explicitly
aware that an eleventh role gets the same scrutiny as the tenth. Read that
spec, not this doc, for the deliberation; this doc cites the outcome.

## The two exposures — deliberately not the same

**Inside the compose network**, the `comfyui` service offers ComfyUI's
**whole native API and its whole output tree** — every endpoint ComfyUI
ships (`/prompt`, `/api/jobs`, `/history`, `/queue`, `/view`,
`/object_info`, `/system_stats`, …) answers on `http://comfyui:8188` to
anything else on that compose network. Nothing about containerizing it
narrows that surface; it is reachable exactly as it would be on a host
port, just without the host port.

**Outward, through the gateway**, only the **job-scoped `/v1/render`
facade** is published — six allowlisted spellings, nothing else:

- `POST /v1/render`
- `POST /v1/render/uploads/image`
- `POST /v1/render/jobs/{id}/cancel`
- `GET /v1/render/jobs/{id}`
- `GET /v1/render/jobs/{id}/artifacts`
- `GET /v1/render/jobs/{id}/artifacts/{name}`

There is **no** outward spelling of `/history`, `/queue`, or a
caller-parameterised `/view`. The gateway **mints its own job id** on submit
and never returns ComfyUI's `prompt_id` — a caller that only ever sees
gateway-minted ids cannot walk ComfyUI's own counter-named history even by
guessing. Job ids live in the gateway process's memory only: a gateway
restart forgets every id it issued, and a pre-restart id is refused
afterward (`render_job_not_found`) — this is fail-closed and deliberate,
not a gap to file against.

### Request-body caps on the write half

Three of those six spellings carry a body, and the gateway buffers a POST
body whole before forwarding it. The render family is the one lane that
hands a caller's bytes to a **writable** upstream endpoint — ComfyUI's
`/upload/image` writes into its input tree — so an unbounded body would
cost gateway memory first and innereye's disk second. Two caps bound it,
enforced **before** the body is buffered:

| knob | default | applies to |
|---|---|---|
| `GATEWAY_RENDER_MAX_WORKFLOW_BYTES` | 1 MiB (`1048576`) | `POST /v1/render`, `POST /v1/render/jobs/{id}/cancel` |
| `GATEWAY_RENDER_MAX_UPLOAD_BYTES` | 32 MiB (`33554432`) | `POST /v1/render/uploads/image` |

Two caps and not one, because the payloads differ by orders of magnitude: a
ComfyUI API-format workflow graph is JSON in the tens of KB, while an input
image is legitimately several MB. A request over its route's cap is refused
**`413 render_payload_too_large`** and the connection is closed — for a
declared `Content-Length`, before a single byte is read off the socket; for a
`Transfer-Encoding: chunked` body, on the first chunk header that would take
the total past the cap, so the oversized payload never lands in memory
either way. The refusal names the knob it broke. Setting a knob to `0`
disables that cap.

These caps are **render-scoped**. No other POST route is capped by them:
chat completions, embeddings, rerank/score and the `/v1/audio/*` multipart
lanes read their bodies exactly as they did before these knobs existed. A
general request-body limit across every lane is a separate change with its
own blast radius and is **not** what this is.

### When ComfyUI answers headers and then dies

A backend that is cold or stopped is refused before any bytes move, with the
honest `503 render_backend_unavailable` + `Retry-After` described under
"Boundary / non-goals". A backend that accepts the connection, sends
response headers and then **resets, stalls past the read timeout, or sends a
truncated body** is the same fact from the caller's side — the render
backend did not answer this request — so it gets the **same** structured 503
rather than a dropped connection. The message names the mid-response case
explicitly, so the 503 never claims a cold backend it did not observe. This
covers every buffered round trip: submit, polling, the artifact index,
cancel and upload.

**No reader should conclude the native ComfyUI surface is reachable
off-box.** It categorically is not: the `comfyui` service declares
`expose: ["8188"]` with **no** `ports:` key, so port 8188 answers only to
other containers on the compose network — never to the host, and never to
anything outside the box. The isolation property is the absent port
publication, not a bind address (see "Rollback", below, for why the
scaffolded compose service binds `0.0.0.0` inside the container rather than
copying the venv script's `--listen 127.0.0.1`).

## Declaration, not metering

ComfyUI has no `--gpu-memory-utilization` equivalent — its footprint is
per-**graph**, not per-server: idle ~0, roughly 32 GiB mid-FLUX-render,
larger still for a video graph. `lobes` therefore gives it **no
`gpu_mem_util` fraction, real or invented** — there is no fixed fraction to
declare truthfully.

Instead, the Spark card profile declares `declared_peak_gib = 31.42` under
`[roles.innereye]`, sourced from the bare-venv baseline measurement: a cold
FLUX turn staged CLIP 9318 MiB + Flux 22700 MiB + VAE 159 MiB = 32177 MiB.

**This number has exactly one consumer: the `exclusive_roles` co-residency
veto in `lobes/profiles/shape_render.py` (`overcommitted_groups`).** Every
reference to `declared_peak_gib` in this repo terminates there — it is
**never summed** with `cortex`'s (or any other role's) `gpu_mem_util`
fraction, and **never compared against the card's total memory**. The veto
itself is not arithmetic: it is an operator-authored `[[exclusive_roles]]`
group (`roles = ["cortex", "innereye"]`, `shapes = ["spark-innereye"]`) plus
a prose `reason` string quoting the measured bare-venv numbers. `lobes init`
on the Spark card refuses to resolve a shape that would host both `cortex`
and `innereye` together, naming the reason and the resolving shape
(`spark-innereye`) in the refusal text — that refusal is the entire
mechanism. **Say this plainly because it is easy to over-read**: `lobes`
declares a known-bad pairing and refuses it. It does not measure, cap, or
account for GPU memory for this tenant in any other way, and no code path
anywhere sums `declared_peak_gib` with anything.

Co-residency — `cortex` and `innereye` sharing one Spark — is explicitly a
**later milestone**, not a v1 goal, and v1 does not attempt to answer it by
budgeting harder. The unified-memory GB10 measured only 7788 MB of
comfy-aimdo GPU RAM headroom against a 124611 MB total pool on the bare
venv — not enough margin to share the box with a co-resident `cortex` lane
at `gpu_mem_util` 0.30–0.58 of that *same* pool, because VRAM total equals
RAM total on this card: there is one pool, not two. The expected answer is
**topology**, not arithmetic — a second Spark hosting `cortex` while this
one hosts `innereye`, the mesh-brain end-state (issue #112, decision 3)
rather than a workaround squeezed into the declared-peak mechanism.

## The output directory: assigned to the operator, not silently unowned

innereye's own adapter holds no filesystem coupling to the ComfyUI host by
design — it reads artifacts back over the API, never by touching
ComfyUI's `output/` or `models/` directories directly, which is what lets
the same client drive a server on another host. `lobes` inherits the disk
growth that design choice implies: **ComfyUI keeps its own counter-named
copy of every artifact under `output/` and states plainly it will never
clean it up** (`SERVER_SIDE_COPY_NOTE` in innereye's `comfyui.py`).

`lobes` ships **no retention or cleanup policy** for that directory. The
bind-mounted, writable `output/` tree *is* the exposure: the operator sees
and manages every artifact directly from the host filesystem, including
ComfyUI's own duplicate copies, and is the party responsible for its
growth. This is a deliberate choice, not an oversight — stated here so it
is not left silently unowned. (Compare `models/`, which is mounted
**read-only** — see "Volumes and ownership" below — so that tree cannot
grow at all from inside the container.)

## Volumes and ownership

- `${COMFY_MODELS:-${HOME:-/root}/comfy/ComfyUI/models}` → the container's
  models path, **read-only**. This follows the same structural argument as
  the llama.cpp GGUF precedent already in the fleet template: a read-only
  mount is the statement that lobes never writes to a 65 GiB tree it did
  not create. **Accepted consequence:** downloading a checkpoint from
  inside ComfyUI is ruled out; the operator provisions weights.
- `${COMFY_OUTPUT:-...}` → the container's output path, **read-write**, and
  separately owned from the read-only models tree.
- The service declares `user: "${COMFY_UID:-1000}:${COMFY_GID:-1000}"` — the
  **first** fleet service to set a `user:` at all. It is load-bearing: no
  fleet Dockerfile carries a `USER` directive, so every container otherwise
  runs as root, and a root-owned artifact under a bind-mounted host tree is
  one the host user cannot delete without `sudo`. With `user:` set, a
  rendered artifact is owned `1000:1000` on the host, deletable and
  overwritable without `sudo`, and a write attempt against the read-only
  `models/` mount is rejected regardless.
- The model path is an env knob with a fallback default, never a literal in
  the packaged template — the same `${HF_CACHE:-...}` shape the fleet
  template already uses for the vLLM lanes' HF cache mount. A box whose
  weights live somewhere other than `$HOME/comfy` sets one env knob; the
  packaged compose file carries no operator-specific absolute path.

## Boundary / non-goals

Following the convention `docs/realtime-pipeline.md`'s own audio surface
set (`docs/realtime-pipeline.md:682-716`), this surface explicitly **does
not**:

- **Implement a start-on-demand lifecycle actuator.** `lobes` owns
  start/stop/status only (`lobes up innereye`, `lobes up innereye --down`,
  `lobes capabilities`) — a request that reaches a cold backend gets an
  honest `503` + `Retry-After` "warming" response, never a silent boot. The
  gateway's request path has no lifecycle actuator to trigger one: no file
  under `lobes/gateway/` imports `lobes.runtime._compose`, and a test
  enforces that absence so a later refactor cannot quietly cross it.
- **Meter or sum GPU memory.** See "Declaration, not metering" above —
  `declared_peak_gib` feeds one boolean co-residency veto and nothing else.
  Nothing in `render.py`, `schema.py`, `shapes.py`, or `shape_render.py`
  sums `gpu_mem_util` (or the declared peak) across roles or compares it to
  a card total, for `innereye` or any other role.
- **Proxy ComfyUI's websocket.** innereye never opens one — its own client
  deliberately polls (`--wait`) rather than consuming `/ws`, because a
  stdlib-only client cannot easily do so — so the render facade has no
  websocket leg to carry, and none was built. (Compare
  `/v1/realtime`, which *does* need a bespoke bridge for its own client;
  that need simply does not exist here.)

## Mesh: discovery without reach

`innereye` is a member of `lobes.roles.MESH_UNFORWARDABLE_ROLES` — it is
**never** auto-wired into this box's own mesh advert/heartbeat, even when
hosted, feasible, and carrying a real fingerprint. This is a **narrower**
carve-out than "every path-routed role": the audio roles (`stt`/`tts`) are
also path-routed but **are** forwardable, because each is a single
POST-in/response-out call the existing forwarder's shape can carry.
`innereye`'s render lane is submit-then-poll-then-GET-binary — three round
trips with server-held state between them — and the gateway's only forward
primitive, `open_upstream`, is single-hop, single-response, and
(pre-existing, before this facade) hardcoded to POST. That shape has no
home in it, mirroring why `/v1/realtime` needed its own bespoke bridge
rather than reusing the same POST forwarder.

Concretely: `GET /capabilities` on the **hosting** box always lists
`innereye` — wired or not — so a peer that has never been told an origin
can still discover the lane by asking that box directly. What the mesh does
**not** do is relay a render request from a box that doesn't host the lane
to one that does. This is claim `c9` from the converged spec, stated
plainly: **the mesh gives discovery for free but not reach.** It is not
advertised-but-unreachable in the sense #92 forbids — a peer is never told
the lane exists on a box it cannot use it from; it is simply never told the
lane exists on a box that isn't hosting it, which is the same honesty
contract every other role follows.

## Readiness: a probe, not a liveness check

ComfyUI has no `/health` route — measured live, it 404s. Every other lane's
readiness probe (`lobes/runtime/_health.py`, `lobes/gateway/_readiness.py`)
hardcodes that literal; `innereye` is the one backend that needs a
different path, plumbed as a per-backend override rather than a change to
the shared default. The render lane probes `GET /object_info` instead,
which measures against ComfyUI 0.33.2 answers 200 — but only *after* the
custom-node import pass completes. A `200` from that probe is therefore
**readiness** (the server has finished loading and can accept a submit),
not mere **liveness** (the process is up and answering something) — a
distinction worth stating because ComfyUI's own `/` and `/system_stats`
answer 200 earlier, during the loading window, and would falsely advertise
`ready: true` if used instead.

## Rollback

The compose service is not the only way to run this tenant. **With the
`comfyui` compose service stopped, `start-comfy.sh` still brings the venv up
against the same model tree** — the containerization did not replace or
disturb the host venv or the weights it points at; it is a second,
independently startable way to run the same install. This is the accepted
rollback path, and it is nearly free precisely because it was never
removed.

**Downtime during cutover is accepted, not engineered around.** innereye
reaches ComfyUI at `http://127.0.0.1:8188` on the host today; once the
`comfyui` service is containerized with no published port, that address
stops answering until innereye's client adopts the facade wire. The two
repos are **not** sequenced to avoid that gap — innereye is in an
experimentation phase and is not being productized, so the operator
decision (claim `c31`) accepted the downtime rather than coordinating a
synchronized cutover. **The facade-adoption issue on
`agentculture/innereye` still needs to be filed** so the client knows its
wire changed (the `lobes`-side scaffolding in this repo does not itself
file that issue — see the note at the end of this doc).

## Deployment catalog and the lock

A `deployments/` catalog entry for a box running this shape, if and when
one is captured, must carry a `VARIATION.md` stating plainly: **restoring
this deployment's lock yields a service *definition*, not a renderable
deployment.** The lock (`deployment.lock.toml`) captures the compose
service, the Dockerfile, and the `[env]` allowlist — but the 65 GiB model
tree bind-mounted into it is **host state no lock carries**, exactly like
the deployment lock's existing exclusions (`_URL`-suffixed keys, `.env`
itself). A box that restores from this lock gets a `comfyui` service that
will start and answer `/object_info`, but cannot successfully submit a
render until the operator separately provisions the model tree at the
mounted path — the lock is silent on how, and that silence is the point:
provisioning 65 GiB of weights is out of scope for a config-restore
mechanism. This doc is where that statement lives for now, following the
`deployments/jetson-agx-thor__thor-worker/VARIATION.md` format's own
"what a restore does and does not give you" convention; a dedicated
catalog entry under `deployments/` is deferred until a real box's lock is
captured (task t13's job, alongside the live acceptance transcript) rather
than written speculatively against a box nothing here has captured yet.

## See also

- `docs/specs/2026-09-16-innereye-lobes-hosts-comfyui.md` — the converged
  spec this doc summarizes; every claim id (`c*`/`h*`) cited above is
  defined there.
- `docs/plans/2026-09-16-innereye-lobes-hosts-comfyui.md` — the build plan,
  including task t13 (the live acceptance run that will convert this doc's
  DECLARED/UNVALIDATED status to validated).
- `docs/evidence/2026-09-16-baseline-comfyui-venv-spark.txt` and
  `docs/evidence/2026-09-16-t2-comfyui-container-spark.txt` — the two
  existing transcripts, covering the ComfyUI workload alone (bare venv and
  containerized), not the gateway-fronted lane.
- `docs/colleague-stack.md` — the eleven-role contract `innereye` now
  appears in.
- `docs/realtime-pipeline.md` — the audio surface, whose "Boundary /
  non-goals" convention this doc follows, and the nearest precedent for a
  non-vLLM sidecar with its own bespoke gateway plumbing.
