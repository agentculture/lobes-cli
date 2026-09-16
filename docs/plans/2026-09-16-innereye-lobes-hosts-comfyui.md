# Build Plan — innereye: lobes hosts ComfyUI

slug: `innereye-lobes-hosts-comfyui` · status: `exported` · from frame: `innereye-lobes-hosts-comfyui`

> lobes hosts ComfyUI as a managed tenant: one command warms the image backend after a reboot, its peak cost is DECLARED against the same card the gears draw from so a known-bad pairing is refused, and a peer on another machine reaches it through the gateway's bearer — while ComfyUI itself publishes no host port and is reachable only on the compose network

## Tasks

### t5 — Register innereye as the eleventh role ATOMICALLY: every role-keyed table, declaration-only

- instruction: Approved deviation d2: registering a role is atomically cross-cutting and cannot be split file-disjoint — the repo's own invariant tests enforce all-or-nothing. This task now absorbs t14 (profiles/schema.py ROLES + `shape_render`.`ROLE_SERVICE`), t7 (goldens), and the stub table entries formerly scoped to t3 and t10. DECLARATION ONLY: stub values, feasible:false, no behaviour. All behaviour stays with t6/t9/t10/t2/t3/t13.
- covers: c15, h2, c6, h15
- acceptance:
  - innereye is in ROLES, `ROLE_BACKEND` and `ROLE_PATH` with hardcoded `_INNEREYE_MODEL`/`_INNEREYE_RUNTIME` constants, and is deliberately ABSENT from `ROLE_ROLE_HINT` so `GATEWAY_FRONTED_ROLES` excludes it — exactly as stt/tts do
  - A dropped or unhosted innereye reports feasible:false on both capabilities surfaces, is omitted from /v1/models, and 404s `role_infeasible` on every alias — never half-served (#92)
  - `ROLE_PATH`\["innereye"\] is the single facade string; no other role's `ROLE_PATH` row changes
  - Every role-keyed table carries an innereye entry and the FULL suite is green: the repo's own all-or-nothing invariants pass (`test_role_service_map`, `test_every_proxyable_role_resolves_a_served_name`, `test_every_config_env_key_reaches_the_gateway_container`, `test_up_no_deps_isolation`\[innereye\])
  - Regenerated goldens are INSPECTED, not merely passing: the diff shows `INNEREYE_`\* keys and the 11th role in the capabilities wire, and nothing else

### t6 — Profile and shape wiring: `INNEREYE_`\* env family and the declared-peak co-residency veto

- instruction: Resist the symmetry pull: the tenant has no `gpu_mem_util` fraction and must not be given a fake one. The veto works by operator declaration plus a prose reason, never arithmetic — that is the mechanism, not a limitation to fix.
- depends on: t5
- covers: c17, h4, c5, h17
- acceptance:
  - The declared peak is a per-card figure whose ONLY consumer is the `exclusive_roles` co-residency veto in `shape_render.py` — grepping its every reference shows no summation and no comparison against card total
  - No budget arithmetic is added anywhere; the render lane's `INNEREYE_`\* keys follow the `ROLE_ENV_PREFIX`/`_KNOB_ENV_SUFFIX` convention in render.py

### t8 — Gateway plumbing: a method-general upstream opener and a GET-side streaming relay

- instruction: This is pure plumbing in lobes/gateway/server.py and lands BEFORE the facade routes (t9) so the two do not collide in the same file. `_relay_streaming` already handles binary correctly — it has simply never been called from a GET route. Stdlib only: no requests/httpx.
- covers: c3, h14
- acceptance:
  - `open_upstream` (server.py:583-620) accepts a method instead of hardcoding conn.request('POST', ...), or a GET-side helper exists alongside it; existing POST callers are unchanged in behaviour
  - `_relay_streaming` is reachable from `do_GET` and relays arbitrary binary chunked bodies; artifact bytes fetched through it are byte-for-byte identical to the same render fetched directly from ComfyUI on the compose network

### t9 — The /v1/render facade: job-scoped routes, fan-out, and the auth placement

- instruction: Mirror how `is_audio_path` routes /v1/audio/\* to a single backend without model-field routing. The bearer is per-branch, not middleware: every POST inherits it, but GET only inherits it under /v1/ (server.py:4426) — that is why the family must be spelled /v1/render. If v1 declines to track issued ids, the doc must instead state shared-history visibility plainly; h28 forbids silence.
- depends on: t8, t5
- covers: c2, h16, c26, h12, c35, h28, c38, h31
- acceptance:
  - The family is spelled under /v1/ and wired into BOTH the `do_GET` and `do_POST` if/elif chains; an unauthenticated GET under /v1/render is refused 401, proven by negative control rather than assumed from the prefix
  - The facade is job-scoped: the gateway issues the job id on submit and serves status and artifacts only for ids it issued; an authorized caller requesting a job id it did not submit is refused
  - The fan-out reaches ComfyUI's submit, poll and artifact-fetch endpoints without exposing /history or /view enumeration outward

### t11 — Capabilities and mesh advert for the render lane

- instruction: Decide explicitly whether v1 announces the lane to the mesh or keeps it off. Note that no operator-facing producer sets 'private' today — `declared_lane_config` (server.py:4981-4994) never emits the key — so 'announce it private' is not currently expressible without adding that producer. The mesh Fingerprint is (`served_id`, quantization, `max_model_len`, runtime); three of the four are meaningless here and would verify by matching unknowns.
- depends on: t5, t9
- covers: c21, h7, c9, h20
- acceptance:
  - A peer that has never been told an origin can discover the render lane from GET /capabilities alone
  - The mesh behaviour is decided and observable: either the lane never appears in another member's roster, or it appears and a cross-box render is demonstrated end to end — never advertised-but-unreachable (#92)

### t12 — The reference doc, the non-goals, and the catalog entry

- instruction: Downtime is accepted (c31) — do NOT sequence the repos to avoid a gap, but DO file the facade-adoption issue on agentculture/innereye so the client knows its wire changed. Per #108 every surface says DECLARED/UNVALIDATED until the t13 transcript lands.
- depends on: t11
- covers: c12, h21, c23, h9, c24, h10, c31, h25, c40, h33
- acceptance:
  - A new reference doc carries an explicit Boundary / non-goals section following docs/realtime-pipeline.md:682-716, naming: no start-on-demand actuator, no metering or summing of GPU memory, no websocket proxy
  - The doc states the TWO exposures side by side — ComfyUI's full native API and output tree inside the compose network, versus only the job-scoped /v1/render facade outward — so no reader concludes the native surface is reachable off-box
  - The doc claims only declaration and refusal, never metering; the output directory's growth is either bounded by a stated policy or explicitly assigned to the operator, not left silently unowned
  - A rollback is stated: with the compose service stopped, start-comfy.sh still brings the venv up against the same tree; and VARIATION.md states that a lock restore yields a service definition, not a renderable deployment

### t1 — Declare the innereye/cortex `exclusive_roles` group and prove lobes init refuses the bad shape

- instruction: v1 never co-resides (c41). This task is the DECLARATION and its guard only — pure card-profile data plus the co-residency test. The aimdo observations moved to t2, which is where a container actually exists to observe.
- depends on: t5
- covers: c41, h34
- acceptance:
  - A \[\[`exclusive_roles`\]\] group declaring roles=\["cortex","innereye"\] with a shapes= list and a reason quoting the measured bare-venv numbers is added to the card profile; lobes init REFUSES a resolved shape hosting both, and the refusal names both the reason and the resolving shape

### t2 — Dockerfile.comfyui: reproduce the working install as a locally-built image

- instruction: Model on Dockerfile.chatterbox: locally built, not pulled; a lean CUDA base; the lobes-cli wheel install pattern. `custom_nodes` carries no third-party nodes, so nothing beyond stock ComfyUI needs reproducing.
- depends on: t1
- covers: c19, h5
- acceptance:
  - The image pins ComfyUI v0.33.2 and installs torch from the stock <https://download.pytorch.org/whl/cu130> index, matching the venv it replaces
  - The built image renders at least one FLUX job, proving the containerized stack matches the bare venv; and the run RECORDS the comfy-aimdo headroom figure, whether DynamicVRAM is still detected, and the pinned-memory line beside the bare-venv baseline (7788 MB / 112149.0) as OBSERVATIONS for the co-residency milestone (r2), never as a v1 gate

### t3 — Compose service skeleton: expose-only isolation, healthcheck, logging, GPU access

- instruction: Reuse the fleet template's existing shapes verbatim: restart: unless-stopped, deploy.resources.reservations.devices for GPU, `env_file` chaining to .env and .secrets.env, mg-logwrap entrypoint. Do NOT copy --listen 127.0.0.1 from start-comfy.sh — in a container that makes the lane unreachable from the gateway.
- depends on: t2
- covers: c28, h22, c29, h23, c8, h19, c36, h29
- acceptance:
  - The service binds --listen 0.0.0.0 --port 8188 in-container and declares expose: \["8188"\] with NO ports: key; a comment on the service carries innereye's loopback rationale forward in meaning, citing finding c36 and explaining that the no-authn property is answered by publishing no host port
  - A bespoke healthcheck probes GET / or GET /`object_info` (ComfyUI has no /health) and reports unhealthy during the model-loading window rather than healthy immediately; the shared /health literal in runtime/`_health.py` and gateway/`_readiness.py` is NOT parameterized
  - Negative controls pass: requests to 8188 from off-box AND from the host are both refused, while the gateway reaches the service as <http://comfyui:8188> on the compose network

### t4 — Compose volumes and ownership: read-only weights, writable output, non-root user

- instruction: The read-only models mount follows the llama.cpp GGUF precedent at docker-compose.yml:321-325 as the structural statement that lobes never writes to a 65G tree it did not create. No fleet service sets user: today — this is the first, and it is load-bearing. Accepted consequence: downloading a checkpoint from inside ComfyUI is ruled out.
- depends on: t3
- covers: c20, h6, c30, h24, c32, h26, c37, h30
- acceptance:
  - Volumes are ${`COMFY_MODELS`:-${HOME:-/root}/comfy/ComfyUI/models}:/opt/ComfyUI/models:ro and a separately-owned ${`COMFY_OUTPUT`:-...}:/opt/ComfyUI/output read-write; the packaged template contains no operator-specific absolute path
  - The service declares user: "${`COMFY_UID`:-1000}:${`COMFY_GID`:-1000}"; a freshly rendered artifact inspected from the host is owned 1000:1000, not root, and the host user can delete it and write into the output tree without sudo
  - A write attempt to the models mount from inside the container is rejected, and rendering the compose on a box whose weights are not under $HOME/comfy needs exactly one env knob

### t10 — Lifecycle: the up target, the cold-backend 503, and the no-actuator guard

- instruction: Start-on-demand is explicitly NOT in v1 (c18). The import-absence test is the load-bearing artifact here: it is what keeps the separation the codebase currently holds absolutely.
- depends on: t5, t3
- covers: c16, h3, c7, h18
- acceptance:
  - lobes up innereye starts and --down stops exactly that service, carrying --no-deps like every other target; lobes capabilities reports whether it is warm
  - A request reaching a cold backend returns an honest warming response with a Retry-After — never a silent boot
  - A test fails on any `_compose` import under lobes/gateway/, so a later refactor cannot quietly put a lifecycle actuator in the data plane

### t13 — Live acceptance run on the DGX Spark and its evidence transcript

- instruction: This is the task that converts every DECLARED surface to VALIDATED. Nothing in docs or lobes capabilities may claim validated before this transcript lands (#108). Include the negative controls — a probe without them is vacuous.
- depends on: t4, t10, t12
- covers: c1, h1, c22, h13, c25, h11, c39, h32
- acceptance:
  - A transcript under docs/evidence/ records all four success signals: exactly 1 lobes command brings the backend warm after a reboot; GET /capabilities reports innereye feasible and ready; an unauthenticated request to the render family is refused 401 while 8188 is refused from both off-box and the host; and at least 1 end-to-end FLUX render completes through the gateway with its artifact bytes returned
  - Two overlapping render submissions are measured: the transcript records whether ComfyUI serialized them and the observed peak, which is what any `MAX_ACTIVE` value rests on
  - The innereye leg is recorded honestly as a cross-repo dependency — a render through the gateway requires innereye to have adopted the facade wire first, so its absence is stated, not silently skipped

## Risks

- [unknown_blocking] CO-RESIDENCY risk (not a v1 risk): comfy-aimdo sizes itself from the whole unified pool — measured on the bare venv as 112149.0 pinned memory and 7788 MB headroom against 'Total VRAM 124611 MB, total RAM 124611 MB' — and its behaviour inside a cgroup is unmeasured. If it reads the host total from within a container, a declared peak is a function of start order rather than a property of the workload, and the c17 mechanism would be the wrong tool. Every one of those outcomes is CONDITIONAL on sharing the card, which v1 forbids (c41)
- [unknown_nonblocking] Co-residency milestone: before innereye is ever hosted alongside a heavy generate lane, the aimdo-in-container numbers must be measured — the headroom figure, whether DynamicVRAM is still detected, and the pinned-memory line, each against the bare-venv baseline (7788 MB / 112149.0). Until then the `exclusive_roles` group stands and no declared peak should be trusted as a co-residency budget
