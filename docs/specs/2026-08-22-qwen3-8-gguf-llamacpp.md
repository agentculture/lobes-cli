# qwen3.8-gguf-llamacpp

> lobes serves unsloth/Qwen3.8-27B-GGUF:UD-`Q4_K_M` via a llama.cpp lane as an alternative engine to the vLLM NVFP4 cortex
> instruction: spike first (TRT-LLM pattern): stand llama.cpp up on the Orin with the UD-`Q4_K_M` GGUF, gate on load+correctness+tok/s+context, THEN integrate (compose lane, catalog entry, shape) only on a GO

## Audience

- the operator of the Orin box and the Culture mesh callers that address model=cortex/main through this box's gateway (plus lobes-cli maintainers who own the catalog/compose/telemetry surfaces)

## Before → After

- Before: cortex on the Orin is proxy-only (`PRIMARY_PEER_PROXY` to the Spark) because the NVFP4 W4A4 cortex quant is Blackwell-only; the box spends its GPU budget hosting senses (Gemma 4 12B), a role the operator is willing to give up
- After: the Orin serves cortex LOCALLY: llama.cpp runs unsloth/Qwen3.8-27B-GGUF:UD-`Q4_K_M` behind the lobes gateway, senses/Gemma 12B is removed, and model=cortex answers on this box without a proxy hop

## Why it matters

- a local cortex removes the Spark dependency for this box's reasoning lane and puts the fleet's `sm_87` silicon to work on the role that matters most — the first cortex the Ampere box can actually run

## Requirements

- catalog.py's Gear model is vLLM-shaped (id doubles as --served-model-name, `tool_parser` is a vLLM --tool-call-parser, `serve_extras` are vLLM flags) — a llama.cpp-served GGUF gear needs either a runtime/engine field on Gear or a lane declared outside the catalog
  - instruction: add an engine/runtime axis (or a parallel llama.cpp lane declaration) such that existing vLLM gears render byte-identically; prove with the goldens/tests before merging
  - honesty: a llama.cpp gear can be declared without breaking the existing vLLM gears: every current catalog consumer (switch, fleet render, `infer_parser`) still resolves the vLLM gears byte-identically after the change
- every generate lane in both compose templates (legacy docker-compose.yml + fleet template) images a vLLM container (NGC 26.04 or the pinned vllm-openai nightly digest) — a llama.cpp lane means a new service block with its own image, and no ARM64/Blackwell (`sm_110`/`sm_121`) llama.cpp CUDA image is verified in-tree
  - instruction: pin a llama.cpp CUDA image or build for JetPack/CUDA 13.2 with runtime: nvidia (csv-mode toolkit — no deploy.resources GPU requests); record the exact image/commit in the compose lane
  - honesty: a llama.cpp CUDA build/image actually runs on this box's JetPack (CUDA 13.2, csv-mode container toolkit — deploy.resources GPU requests fail here, the lane must use runtime: nvidia)
- the GGUF cortex REPLACES senses on this box (operator decision 2026-08-23: Gemma 4 12B is removed): the Orin becomes a cortex-lobe — llama.cpp Qwen3.8-27B UD-`Q4_K_M` (~16-17 GB weights + KV) plus the pooling gears, with senses' ~0.45-util budget reclaimed
  - instruction: stop the vllm-multimodal lane, then boot the llama.cpp cortex and measure RSS/unified-memory at target context on this swapless box before declaring the budget
  - honesty: the UD-`Q4_K_M` weights + KV cache + pooling gears fit the 61.3 GiB unified budget with NO swap configured, measured at the target context, not computed
- the rollout must neutralize the known Tegra spurious-iowait shedding on this box, or the local cortex is unusable during flares: sugov kthreads inflate `nr_iowait` with zero disk I/O, and the gateway pressure policy (iowait > 50 -> busy, lobes/gateway/`_pressure_policy.py`:166) 429-sheds ALL cortex traffic; the deployed .env still carries threshold 50 and the =100 override is ephemeral (container shell-env, reverts on next compose up)
  - honesty: verified on the box after cutover: with the persisted threshold (or Tegra-aware sampling), a sugov iowait flare no longer flips the gateway to busy — cortex requests still answer 200 during a flare, and the setting survives a docker compose down/up cycle
- the llama.cpp lane lands as rendered template/profile/shape data, NEVER as hand-edits to the deployed compose: this box already carries hand-edited runtime: nvidia lines that a re-init would revert (recorded 2026-07-16 drift), and orin.toml:48 documents that debt — the new lane must not deepen it
  - honesty: a fresh lobes init render for this box produces the llama.cpp lane with runtime: nvidia and every csv-mode accommodation from repo data alone — zero manual compose edits needed to boot
- staged cutover with a live rollback path: `PRIMARY_PEER_PROXY` to the Spark stays configured and working until every c20 gate passes on the local lane, and senses is stopped only when the llama.cpp cortex needs its memory — never removed as step one; rollback is re-enabling the proxy, which requires no peer-side change
  - honesty: at every step of the cutover, model=cortex through this gateway answers correctly (proxied or local), and the flip back to proxy is a .env-only change proven once before senses is removed

## Honesty conditions

- llama.cpp's server really does serve this hybrid-Mamba Qwen3.8 GGUF with an OpenAI-compatible /v1/chat/completions on `sm_87` — proven by a live spike on this box, not by hearsay
- a live request to model=cortex through this box's gateway reaches the llama.cpp backend and returns a well-formed OpenAI chat completion with zero gateway code changes
- lobes assess runs green against the llama.cpp lane unmodified — if any probe needs a llama.cpp special-case, this boundary is false and must be re-scoped
- mesh callers keep addressing role aliases (model=cortex/main), never a llama.cpp-specific id — the engine swap is invisible at the calling contract
- the before-state is verified on the deployed box, not from memory: the .env shows `PRIMARY_PEER_PROXY` to the Spark and vllm-multimodal is the running senses lane at the time the swap starts
- after the swap, model=cortex answers locally (no X-Lobes-Proxied-By header) and the Gemma 12B container is down and its budget freed
- the local cortex is actually usable for real work: latency and quality are acceptable to the operator after a hands-on session, not just probe-passing
- assess numbers are captured through the gateway path callers actually use, and the evidence transcript lands under docs/evidence/ before any doc claims VALIDATED (#108 rule)
- the numbers come from the live box run, measured through the gateway path, and land in the docs/evidence/ transcript — no target is claimed met from theory or extrapolation
- docker ps on the deployed box shows no host-published port for the llama.cpp container, and the backend answers only via the gateway origin — probed from another mesh box

## Success signals

- lobes assess passes against the llama.cpp lane through the gateway (health, known-answer, tool-calling probe), with measured decode tok/s and a working context window recorded in an evidence transcript under docs/evidence/, per the house pattern
  - instruction: run lobes assess + the tool-calling probe through the gateway, capture decode tok/s and context ceiling, land the transcript under docs/evidence/ with the date-stamped naming convention
- measured on this box through the gateway: decode >= 5 tok/s single-stream, context window >= 32768 served and needle-probed, known-answer and tool-calling probes PASS — numbers recorded in the evidence transcript

## Scope / boundaries

- the gateway request path stays untouched: Backend keys on `base_url` and forwards OpenAI-shaped POSTs verbatim, so any OpenAI-speaking llama.cpp server drops in behind the existing routing (established by docs/tensorrt-llm-investigation.md section 1, re-verified against lobes/gateway/)
- lobes assess / lobes benchmark need no change: assess.py speaks pure OpenAI HTTP (/health, /v1/models, /v1/chat/completions), which llama.cpp's server implements — so llama.cpp-vs-vLLM numbers are directly comparable through the existing harness
- the llama.cpp backend joins the fleet like every vLLM lane: unpublished port on the compose network, reached only through the gateway (vllm-multimodal precedent — no host port), so the gateway's opt-in `GATEWAY_API_KEY` gate remains the sole auth surface

## Non-goals

- this is not a vLLM replacement decision: docs/tensorrt-llm-investigation.md set the house pattern for alternative engines — desk/spike investigation with a go/no-go on bring-up effort BEFORE any deployment claim; the GGUF lane follows the same pattern and the NVFP4 vLLM cortex stays the deployed primary until measured otherwise
- mesh senses re-homing and the Spark's dangling senses-proxy wiring are OUT of this frame's scope — the operator accepts losing Gemma 12B entirely if needed; this frame delivers only the Orin-local GGUF cortex

## Assumptions

- gateway /status and lobes overview --live will report misleading zeros for a llama.cpp backend: lobes/`_metrics.py` parses vllm:\* Prometheus series only (vllm:`gpu_cache_usage_perc`, vllm:`num_requests_running`, ...) — a llama.cpp metrics adapter or an honest 'telemetry unsupported' marker is needed
- the GGUF lane loses known vLLM-lane features until proven otherwise: `preserve_thinking` chat-template kwarg (#93), strict tools via xgrammar structural tags (colleague#320), the qwen3 reasoning parser, self-hosted MTP speculative decoding, and multimodality (a text GGUF has no ViT unless an mmproj file ships and llama.cpp wires it) — each is a per-feature parity check, not assumed carried over
- a second cortex host raises a routing question the current mesh has no precedent for: today each role has at most one hoster and peer referral/proxy names a single origin — an Orin-local GGUF cortex either serves only local callers or needs the shape/referral story extended
- the WHY: this Orin's operator profile (~/.lobes/profiles/orin.toml, 2026-07-16 mesh work) declares cortex feasible=false because modelopt NVFP4 W4A4 needs Blackwell — GGUF `Q4_K_M` via llama.cpp is the quantization path Ampere `sm_87` CAN run, replacing today's cortex-by-proxy-to-Spark with a local cortex
- removing senses from the Orin leaves the mesh with NO senses host unless one is re-declared elsewhere: the Spark currently proxies model=senses to this box (live 2026-07-31 wiring), so the Spark's SENSES/MULTIMODAL peer config dangles and vision intake mesh-wide needs a decision — re-host, referral-only, or accept `role_infeasible`
- thinking-mode response-shape parity is unverified: the vLLM lane parses <think> into `reasoning_content` via --reasoning-parser=qwen3, while llama.cpp uses its own --reasoning-format mechanism — whether callers see `reasoning_content` (vs thinking leaking into content, or silently stripped) on the llama.cpp lane must be checked in the spike

## Scope exploration

- `s1` — `lobes/catalog.py`: Gear dataclass fields are vLLM flags verbatim (served-model-name, `tool_parser` matching runtime.`_parser`.`infer_parser`, MoE serve extras); no engine/runtime axis exists
  - seeds: `c2`
- `s2` — `lobes/templates/docker-compose.yml + templates/fleet/docker-compose.yml`: all 10+ generate-lane image: lines are vLLM images; zero llama.cpp/GGUF mentions anywhere in the repo (grep across py/md/toml/yml returned nothing) — greenfield runtime surface
  - seeds: `c3`
- `s3` — `lobes/gateway/ (via docs/tensorrt-llm-investigation.md sec.1)`: request proxying is engine-agnostic; llama.cpp's OpenAI-compatible server is reachable behind the gateway unchanged
  - seeds: `c4`
- `s4` — `lobes/_metrics.py`: telemetry parser is keyed on vllm:\* series names; llama.cpp's /metrics surface differs — same gap the TRT-LLM investigation recorded
  - seeds: `c5`
- `s5` — `lobes/assess.py`: assessment harness is stdlib OpenAI HTTP only; llama.cpp serves /health and /v1/models, so the harness runs as-is
  - seeds: `c6`
- `s6` — `docs/tensorrt-llm-investigation.md`: prior alternative-engine precedent: engine-agnostic request path confirmed, telemetry gap named, verdict pattern = spike before bring-up
  - seeds: `c7`
- `s7` — `docs/qwen3.8-27b-nvfp4.md (checkpoint facts)`: the NVFP4 lane's validated features (MTP module, ViT, `preserve_thinking` template, `qwen3_coder_thinking` strict-tools plugin) are vLLM mechanisms; none transfer to llama.cpp automatically
  - seeds: `c8`
- `s8` — `lobes/profiles/builtin/orin.toml + builtin_shapes/orin-lobe.toml, orin-small.toml`: the Orin card has a profile and two shapes in-tree; orin-small is declared-UNVALIDATED (no 27B, senses disabled per base fallback rules) and this box actually runs senses — a GGUF-cortex-on-Orin shape is a new shape, not a tweak of either
  - seeds: `c9` (rejected), `c10`
- `s9` — `~/.lobes/profiles/orin.toml + this box's deployed .env (cortex=proxy to Spark)`: cortex is currently infeasible-local on `sm_87` (NVFP4 W4A4 is Blackwell-only) and served by `PRIMARY_PEER_PROXY` to the Spark; the GGUF lane would make it local. Budget note: box ran 54/61 GiB with 3 engines and NO swap — a ~16-17 GB `Q4_K_M` plus KV must displace or coexist within that
  - seeds: `c11`, `c9` (rejected)
- `s10` — `CLAUDE.md proxy-lobes section + this box's deployed compose (vllm-multimodal lane)`: the Spark's gateway forwards senses to this Orin (X-Lobes-Proxied-By validated 2026-07-31); dropping the 12B here breaks that chain — the removal is a mesh topology change, not a local free-up
  - seeds: `c13`
- `s11` — `challenge pass / adjacent-systems lens: lobes/gateway/_pressure_policy.py + this box's deployed .env`: an Orin-local cortex inherits the gateway pressure gate; the spurious-sugov-iowait flare (memory, 2026-07-17, validated live) would shed it exactly as it shed senses — persist the threshold or land Tegra-aware sampling as part of this work
  - seeds: `c21`
- `s12` — `challenge pass / unstated-assumptions lens: docs/qwen3.8-27b-nvfp4.md reasoning sections vs llama.cpp server surface`: the frame gates tool calling (c20) but never mentions the reasoning-trace response shape callers get today; parity is a spike check, not an assumption to carry silently
  - seeds: `c22`
- `s13` — `challenge pass / lifecycle lens: lobes/profiles/builtin/orin.toml:48 + deployed ~/.lobes compose hand-edits`: re-init reverts hand edits; the csv-mode runtime: nvidia requirement (h3) must be rendered from repo data so lobes init on this box produces a working lane byte-for-byte
  - seeds: `c23`
- `s14` — `challenge pass / reversibility lens: deployed .env (PRIMARY_PEER_PROXY wiring) + shape restore mechanics`: the frame said replace-senses but never ordered the swap; keeping the proxy as fallback makes the cutover reversible at every step
  - seeds: `c24`
- `s15` — `challenge pass / security lens: templates/fleet compose port exposure + gateway auth`: no new auth surface: backend stays network-internal like the existing lanes
  - seeds: `c25`
- `s16` — `challenge pass / concurrency lens: llama.cpp server slot model vs mesh caller patterns`: single-slot default queues concurrent requests; acceptable for a one-operator box, unmeasured beyond that — parked nonblocking
- `s17` — `challenge pass / cheap probes: HF API + df + free + /proc/stat on this box`: UD-`Q4_K_M` exists upstream at 16.46 GB (HF tree API, live probe); 1.6 TB disk free; box at 37/61 GiB with senses up, swap 0; iowait 0 right now — the sugov flare is episodic, not steady-state

## Open parks

- [unknown_nonblocking] llama.cpp serves one model with N parallel slots; mesh callers can issue concurrent cortex requests — queueing/latency behaviour under concurrency is unmeasured (single-stream is the only c20 target)

## Resolved vagueness

- [unknown_blocking] llama.cpp support quality for the Qwen3.8 `qwen3_5` hybrid Mamba/linear-attention arch is hearsay ('I hear llama.cpp supports it well') — unverified: whether UD-`Q4_K_M` loads, decodes correctly, supports YaRN to beyond-native context, and at what tok/s on this fleet's aarch64 Blackwell silicon needs a live spike — resolved: operator-confirmed frame gates the whole build on a live spike (c1 instruction + h1): llama.cpp on this box must prove load/correctness/tok/s/context on the UD-`Q4_K_M` before any integration lands — the unknown is resolved by that gate, not by assuming support
- [unknown_blocking] no verified llama.cpp CUDA container/build for aarch64 `sm_110` (Thor) / `sm_121` (Spark GB10) is known in-tree; build-from-source vs ghcr.io/ggml-org/llama.cpp server-cuda image viability is unexplored — resolved: target corrected from Blackwell to `sm_87` (Jetson AGX Orin, Ampere): llama.cpp CUDA on Orin is a mature, well-trodden path — the remaining unknown is only which build/container to pin, folded into the v1 spike
