# lobes

`lobes` is the tooling that **runs, assesses, and switches** the local,
OpenAI-compatible vLLM model the Culture mesh consumes. The binary is `lobes` —
`lobes switch`, `lobes assess`, `lobes serve`, and so on. (`model` still works as
a deprecated alias for `lobes`.)

The served model is what the [`lobes`](#lobes-is-also-the-deployed-agent)
agent connects to over the `acp` `vllm-local` provider. The tool and the deployed
agent share one identity: the same lobes runs the engine and consumes it.

Sibling to [`culture`](https://github.com/agentculture/culture) (the agent mesh),
[`daria`](https://github.com/agentculture/daria) (awareness), and
[`steward`](https://github.com/agentculture/steward) (alignment).

## Install

```bash
uv tool install lobes-cli
```

> `model-gear` on PyPI is a deprecated alias of `lobes-cli` and will continue to
> work, but new installs should use `lobes-cli`.

## Usage

```bash
lobes init --apply          # scaffold a deployment dir (default $HOME/.lobes)
lobes serve --apply         # start the vLLM server (alias: start)
lobes switch nvidia/Qwen3-32B-NVFP4 --apply   # switch the served model
lobes switch nvidia/Qwen3-32B-NVFP4 --purpose decode-heavy --machine spark --apply  # ...in a tuned gear
lobes status                # current model, container state, /health
lobes assess                # correctness probes (markdown for a per-model doc)
lobes benchmark             # decode throughput + prefill latency (shape follows --purpose)
lobes stop --apply          # stop the server

lobes overview              # tool snapshot + served model + candidate list
lobes whoami                # tool, machine, served model, container health
lobes explain switch        # markdown docs for a topic
lobes doctor                # diagnose docker / compose / .env / health
```

Every command supports `--json`. **Write verbs (`switch`, `serve`, `stop`,
`init`) are dry-run by default** and require `--apply` to commit — agents call
CLIs in loops, so safe-by-default is mandatory.

> The `model` command still works as a deprecated alias for `lobes` — existing
> scripts and config files do not need to be updated immediately.

## Running the model locally (vLLM)

`lobes init` scaffolds a deployment directory (default `$HOME/.lobes`) from the
packaged templates. **Since issue #69, bare `lobes init` (no flags) scaffolds
the fleet duo by default** — see ["Running the model behind a gateway
(fleet)"](#running-the-model-behind-a-gateway-fleet) below; that is almost
certainly what you want. This section instead walks through the **legacy
single-model** scaffold: one vLLM server, no gateway, opted into with
`--single` (alias `--legacy`) — a `docker-compose.yml` that stands up the
vLLM model directly as an OpenAI-compatible server on `:8000` (the
*container's own* port; there is no gateway in front of it in this mode —
see the port note in the fleet section below), plus a `.env`. Tuned for DGX
Spark (GB10 Grace Blackwell, 128 GB unified memory) per
[build.nvidia.com/spark/vllm](https://build.nvidia.com/spark/vllm).

Prerequisites: the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
and `docker login nvcr.io` with an [NGC API key](https://org.ngc.nvidia.com/setup/api-key)
to pull the `nvcr.io/nvidia/vllm` image.

```bash
lobes init --single --apply # writes $HOME/.lobes/{docker-compose.yml,.env} (legacy single-model)
# edit $HOME/.lobes/.env to set HF_TOKEN if the model repo is gated
lobes serve --apply         # first run downloads ~28 GB of weights (the 27B primary)
lobes status                # waits/reports until /health is up
```

Verify it is up:

```bash
curl -fsS http://localhost:8000/health
curl -s http://localhost:8000/v1/models   # what's WARM now (the served model), not the catalog
```

Tunables live in the deployment `.env` (`VLLM_MODEL`, `VLLM_GPU_MEM_UTIL`,
`VLLM_MAX_MODEL_LEN`, `HF_CACHE`, …). `VLLM_SERVED_NAME` must match the part
after `vllm-local/` in `culture.yaml` — `lobes doctor` checks this. `lobes
switch` rewrites these keys for you.

### Tuning the gear (purpose + machine)

`lobes switch` resolves the serve config from three layers — the **machine**
profile (`--machine`, default auto-detected: GPU-memory fraction, context,
attention backend), the **workload** profile (`--purpose`, default `balanced`:
the batching knobs and the shape `lobes benchmark` exercises), and the model's
catalog entry (quantization, tool parser). Explicit `--max-model-len` /
`--gpu-mem-util` flags override the machine defaults.

```bash
lobes switch sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP --purpose decode-heavy --machine spark --apply
lobes benchmark --purpose decode-heavy   # shape defaults to the configured VLLM_PURPOSE
lobes explain tuning                     # the full layering
```

Purposes: `balanced` (≈1K in/1K out), `prompt-heavy` (≈8K in/1K out),
`decode-heavy` (≈1K in/8K out). Machines: `spark` (load-tested), `thor`,
`blackwell`, `generic` (configured). The throughput flags and these shapes follow
shahizat's cross-machine NVFP4 benchmark — see
[`docs/tuning-profiles.md`](docs/tuning-profiles.md) and
[`docs/machine-profiles.md`](docs/machine-profiles.md).

### Supported hardware

lobes is load-tested on DGX Spark (GB10) and Jetson AGX Thor. Unknown cards
receive conservative defaults (small model, no multimodal). See
[`docs/machine-profiles.md#support-table`](docs/machine-profiles.md#support-table)
for the full table and Thor's validated knob divergences.

| card | profile | status | validation |
|---|---|---|---|
| **DGX Spark** (Grace Blackwell, 128 GB) | `spark` | load-tested | 2026-06-03 — fleet duo (cortex 128K, senses 32K) at ~7.8–8.0 tok/s decode with FlashInfer. Correctness probes postdate that run; unverified on the GB10 (#106). |
| **Jetson AGX Thor** (sm_110, 128 GB) | `thor` | load-tested | 2026-07-13 — cortex/embed/rerank correctness probes pass with four validated divergences (kv_cache_dtype/attention_backend/enforce_eager knobs); concurrent first boot needs the boot-ordering caveat. See the Thor section in [`docs/machine-profiles.md`](docs/machine-profiles.md). |
| unknown card | `base` | conservative fallback | — small 4B model, no multimodal, to avoid OOM on first boot. |

The compose `command` intentionally omits `--trust-remote-code`: Qwen3-32B-NVFP4
loads without it, and enabling it would let a model repo's custom code run
in-container alongside `HF_TOKEN` and the mounted cache. Add it back only for a
model whose repo ships custom modeling code. If vLLM rejects the `nvidia/`
ModelOpt checkpoint, set `VLLM_MODEL` to the vLLM-native `RedHatAI/Qwen3-32B-NVFP4`
and drop `--quantization` from the compose `command`.

## Expose the API from anywhere (Cloudflare Tunnel)

`lobes tunnel` publishes the local OpenAI-compatible API at an owner-chosen
hostname through a [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/),
so Culture/AgentCulture agents can call it from anywhere as an ordinary provider
(`base_url` + `api_key`) — no inbound ports, no static IP. The hostname and the
run-token never live in committed config.

> ⚠️ **Gate it first.** A tunnel makes the model reachable from the public
> internet. Set `CULTURE_VLLM_API_KEY` in `$HOME/.lobes/.env` **before** running
> `lobes tunnel` — vLLM then requires `Authorization: Bearer $CULTURE_VLLM_API_KEY`
> on every request. Empty leaves the API open; that is only safe for local dev.
> Generate or rotate the key with `python3 scripts/gen-api-key.py` (writes it to
> the gitignored deployment `.env`; `--show` to print, `--force` to rotate), then
> `lobes serve --apply` to enforce it. You can also set `VLLM_SERVED_NAME` to a
> generic alias (e.g. `default`) to keep the backend checkpoint name out of the
> public `GET /v1/models`.

Two steps — the Cloudflare side once, then the local side:

```bash
# 1) Cloudflare side, ONCE — provision the tunnel + ingress + DNS and seal the
#    run-token in shushu (tunnel-only mode; the backend authenticates itself):
cultureflare remote-login setup \
  --hostname your-host.example \
  --service http://127.0.0.1:8000 \
  --no-access --shushu --apply

# 2) Local side — copy the scaffolded example, fill in hostname + token source:
cp $HOME/.lobes/cf-tunnel.env.example $HOME/.lobes/.cf-tunnel.env
# edit $HOME/.lobes/.cf-tunnel.env (it is gitignored — never commit it):
#   CULTURE_VLLM_PUBLIC_HOSTNAME=your-host.example
#   CULTURE_CF_TUNNEL_TOKEN_SHUSHU=<shushu-secret-name>

lobes serve --apply         # serve (with CULTURE_VLLM_API_KEY set in .env)
lobes tunnel                # DRY RUN: prints the cloudflared command + public URL
lobes tunnel --apply        # start the tunnel in the background
# ... later:
lobes tunnel --stop --apply # tear it down
```

The hostname resolves from `--hostname` → `$CULTURE_VLLM_PUBLIC_HOSTNAME` →
`CULTURE_VLLM_PUBLIC_HOSTNAME` in `.cf-tunnel.env`; the run-token from
`CULTURE_CF_TUNNEL_TOKEN_SHUSHU` (a shushu-sealed secret name, preferred) or
`CULTURE_CF_TUNNEL_TOKEN` (plaintext fallback). `--apply` preflights that
`cloudflared` (and `shushu`) is on PATH and that the local server answers
`/health` first. `cloudflared` + `shushu` are runtime deps on the serving box.

Call it from anywhere — use your hostname and the alias you served (placeholders
shown):

```python
from openai import OpenAI

client = OpenAI(base_url="https://your-host.example/v1", api_key="$CULTURE_VLLM_API_KEY")
client.chat.completions.create(model="default", messages=[{"role": "user", "content": "hi"}])
```

```bash
curl -s https://your-host.example/v1/chat/completions \
  -H "Authorization: Bearer $CULTURE_VLLM_API_KEY" \
  -d '{"model":"default","messages":[{"role":"user","content":"hi"}]}'
```

**Hardening (future).** The bearer key is the minimum bar. For stronger exposure,
layer Cloudflare Access (SSO/service tokens), a WAF rule or IP allowlist, and/or
mTLS in front of the tunnel. Bearer auth currently gates the single-model
deployment; the fleet gateway is not yet auth-aware (planned). See `lobes explain
tunnel` for the full flow.

## Running the model behind a gateway (fleet)

**This is the default scaffold since issue #69** — plain `lobes init --apply`
(no flags) already gives you this; `--fleet` is a back-compat no-op kept only
so old scripts that passed it keep working, and `--single` (alias `--legacy`)
is what opts you *out*, back to the one-container deployment the previous
section describes. `lobes init` scaffolds a **multi-container** deployment:
the always-warm Qwen generate primary, the Gemma 4 12B multimodal gear, two
tiny co-resident **embedding** and **reranker** gears, and a single stdlib
**gateway** that fronts them all on the host port the acp `vllm-local`
provider already expects — `VLLM_PORT` (packaged default `:8000`; `:8001` on
the reference DGX Spark deployment, deliberately set apart from the
single-model story's default). The gateway routes each request by its
`model` field — to the primary, the multimodal gear, the embedder, or the
reranker by task family (generate / embed / score / rerank) — and defaults
an unknown/missing name to the primary, so existing single-model clients
keep working unchanged. The same front fans `/v1/audio/*` out to the
`--audio` overlay, and a warm *generate* fallback can be wired in later (the
gateway adds it, with failover, only when one is configured).

> **`:8000` means two different things depending on scaffold — this
> ambiguity is what issue #92 was about.** In the legacy single-model
> section above, `:8000` is the vLLM **container's own** port, published
> straight to the host with no gateway in front of it. Here, `:8000` (or
> whatever `VLLM_PORT` is set to) is the **gateway's** published port,
> fronting several backends by `model` field. They happen to share the same
> default number, but a client must dial whichever origin its deployment
> actually publishes — never assume a bare port number implies a particular
> topology. `lobes doctor` and `GET /capabilities` are the source of truth
> for what a given deployment actually serves and where.

```bash
lobes init --apply                # $HOME/.lobes/{docker-compose.yml,.env,Dockerfile.gateway} — the default
docker login nvcr.io              # NGC API key for the vLLM image
lobes fleet up --apply            # builds the gateway image + starts the backend
lobes fleet status                # container states + gateway /health + /v1/models
```

```bash
curl -s http://localhost:8000/v1/models       # the WARM backend(s) (not the full catalog — see below)
# an unknown/missing model defaults to the primary; route explicitly by name:
curl -s http://localhost:8000/v1/chat/completions -d '{"model":"sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP","messages":[...]}'
```

The fleet runs a **default-on `cortex` + `senses` duo** (the `main` + `multimodal`
backends) — the 27B Qwen text generate primary served at **128K** (`cortex`, util
`0.30`) and the Gemma 4 12B vision gear served at **32K** (`senses`, util
`0.14` — provisional pending live validation; `coolthor/gemma-4-12B-it-NVFP4A16`,
native MTP default-on — the coder fine-tune, `sakamakismile/gemma-4-12B-coder-…`,
is kept as an opt-in `multimodal-coder` gear; see
[`docs/vllm-nightly-migration.md` §7](docs/vllm-nightly-migration.md))
— plus the tiny embedding + reranker gears (`0.06` each), for a default budget of
`0.30 + 0.14 + 0.06 + 0.06 = 0.56` on the 128 GB GB10. The 4B
`minor` companion and the legacy 14B Qwen are opt-in compose profiles
(`COMPOSE_PROFILES=minor` / `COMPOSE_PROFILES=middle`). Callers address the
generate lane by role/tier alias — `model=cortex|senses` (or
`main|minor|multimodal`; back-compat `hard|cheap|normal`); see
[`docs/colleague-stack.md`](docs/colleague-stack.md) for the six-role contract. `lobes switch` drives the single-model deployment (it can
also serve an embed/score gear solo — auto-detected from the catalog, or forced
with `--task embed|score`); change the fleet primary by editing the fleet `.env`
and re-running `lobes fleet up --apply`. See `lobes explain fleet` / `lobes
explain gateway` for the routing semantics,
[`docs/qwen3-embedding-0.6b.md`](docs/qwen3-embedding-0.6b.md) +
[`docs/qwen3-reranker-0.6b.md`](docs/qwen3-reranker-0.6b.md) for the pooling
gears, [`docs/gemma-4-12b-nvfp4.md`](docs/gemma-4-12b-nvfp4.md) for the
multimodal gear, [`docs/gateway-fleet.md`](docs/gateway-fleet.md) for the
full topology, and [`docs/colleague-stack.md`](docs/colleague-stack.md) for
the six-role Colleague contract (`cortex`/`senses`/`embedder`/`reranker`/`stt`/`tts`,
`lobes capabilities`, `GET /capabilities`).

### Per-model notes

Each runtime model has a doc under `docs/` recording how to run it, live test
results, and caveats:

- [`docs/qwen3.6-27b-text-nvfp4-mtp.md`](docs/qwen3.6-27b-text-nvfp4-mtp.md) — the
  **current** runtime model and fleet default primary
  (`sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`), the 27B re-exported with its MTP
  draft head restored so vLLM speculative decoding (Multi-Token Prediction) works.
  Text-only (ViT vision tower removed), NVFP4 (`modelopt`); served with
  `--tokenizer=mmangkad/Qwen3.6-27B-NVFP4`. Verified 2026-05-31: ~2.4x
  single-stream decode (8 → ~19 tok/s) at ~72-79% MTP draft acceptance, tool
  calling + reasoning confirmed.
- [`docs/qwen3.6-27b-nvfp4.md`](docs/qwen3.6-27b-nvfp4.md) — the **archived former
  primary** (`mmangkad/Qwen3.6-27B-NVFP4`), default primary 0.10.0–0.14.0; load-tested
  on DGX Spark (~8 tok/s decode, ~7 min warm-up). Kept as a candidate: it is the
  tokenizer source the MTP primary serves with and the only vision-capable 27B (the
  MTP primary is text-only).
- [`docs/qwen3-32b-nvfp4.md`](docs/qwen3-32b-nvfp4.md) — the dense **candidate**
  (`nvidia/Qwen3-32B-NVFP4`), faster on decode (~9.7 tok/s); swap in via
  `PRIMARY_MODEL` / `lobes switch` when throughput matters more than context/vision.
- [`docs/gemma-4-12b-nvfp4.md`](docs/gemma-4-12b-nvfp4.md) — the fleet's
  **`multimodal` (normal) tier** (`coolthor/gemma-4-12B-it-NVFP4A16`),
  default-on alongside the primary (issue #69). A unified multimodal checkpoint
  (text+image+audio) with native MTP wired ON by default (28.6 tok/s @ 57.9%
  draft acceptance — see `docs/vllm-nightly-migration.md` §7); replaces the
  demoted 14B as the `normal`/`multimodal` generate tier. The coder fine-tune
  (`sakamakismile/gemma-4-12B-coder-…`) is kept as an opt-in `multimodal-coder`
  candidate — coding-strong, but its MTP acceptance (30.8%) wasn't worth wiring.
- [`docs/mistral-small-3.2-24b-nvfp4.md`](docs/mistral-small-3.2-24b-nvfp4.md) —
  the dense **fallback candidate** (`RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4`);
  the default fleet's warm fallback in 0.11.0–0.19.x, since removed (two ~30B
  NVFP4 models don't co-fit a shared GB10). Kept selectable: load-tested
  2026-05-30, loads reliably (~15 GiB, ~14.9 tok/s decode), text + tool calls.
  Wire it back via the opt-in `FALLBACK_*` fleet config.
- [`docs/qwen3.6-35b-a3b-nvfp4.md`](docs/qwen3.6-35b-a3b-nvfp4.md) — the former
  **MoE fallback** (`mmangkad/Qwen3.6-35B-A3B-NVFP4`), now a candidate. It does
  **not** load reliably on a GB10 shared with other services, and two ~30B models
  do not co-reside there — see [`docs/gateway-fleet.md`](docs/gateway-fleet.md).

The numbers in each doc come from `lobes switch <model> --apply` then `lobes
assess` (correctness) and `lobes benchmark` (throughput). `lobes overview --list`
lists the catalog (these models) and flags which one is currently served.

**Engine investigations** (alternatives to the vLLM serving path) are recorded
separately:

- [`docs/tensorrt-llm-investigation.md`](docs/tensorrt-llm-investigation.md) —
  desk study (2026-06-26) of serving the MTP 27B primary with **TensorRT-LLM**
  (`trtllm-serve`) on the DGX Spark instead of vLLM. **Verdict: not yet** — MTP
  spec-decode is DeepSeek-only in stable TRT-LLM and the Qwen3.6 hybrid GDN
  kernels are RC-only; revisit on TRT-LLM 1.3.0 stable. The lobes request path
  (gateway routing + `lobes assess`/`benchmark`) is already engine-agnostic, so
  re-evaluation stays cheap; only the `/status` `vllm:*` metrics adapter and the
  catalog/switch/template seam are engine-specific.

The two **audio backends** are fixed (the `--audio` overlay, *not* switchable gears
in the catalog), each with its own doc:

- [`docs/parakeet-stt.md`](docs/parakeet-stt.md) — **Parakeet** TDT 0.6B
  (`nvidia/parakeet-tdt-0.6b-v2`, NVIDIA NeMo ASR), the speech-to-text backend
  (`POST /v1/audio/transcriptions`).
- [`docs/chatterbox-tts.md`](docs/chatterbox-tts.md) — **Chatterbox** (Resemble AI,
  0.5B, Apache-2.0), the text-to-speech backend (`POST /v1/audio/speech`), with
  zero-shot voice cloning. Replaced the retired Magpie NIM (no NGC key needed).

### What's loaded vs. what's supported

Two questions that look alike but aren't:

- **What's supported (what can I warm up)?** — the curated catalog of "gears"
  lobes knows how to serve, each tagged `load-tested` (proven on this box) or
  `configured` (declared, not yet proven). It's **static** — defined in
  `lobes/catalog.py`, shipped in the wheel, unchanged by what's running.
  Read it with `lobes overview --list` or the gateway's `GET /v1/models/supported`.
- **What's loaded right now?** — the model(s) actually in GPU memory this instant
  (one in single-model mode; in the fleet, the generate primary plus the
  co-resident embedding + reranker gears). The live source is `GET /v1/models`
  (OpenAI-standard); `lobes fleet status` queries it. `lobes status` /
  `lobes whoami` instead report the model the deployment is *configured* to serve
  (from `.env`) plus container health — normally the same model, but it's
  configuration (which can be stale), not a live query.

| Question | CLI | HTTP |
|---|---|---|
| What *can* I run? (catalog) | `lobes overview --list` | `GET /v1/models/supported` |
| What's *loaded* right now? | `lobes fleet status` | `GET /v1/models` |
| What's the deployment *set* to serve? | `lobes status` / `lobes whoami` | — |

Mnemonic: the catalog is *what's on the menu (and which dishes we've cooked)*;
`/v1/models` is *what's hot now*. See
[`docs/gateway-fleet.md`](docs/gateway-fleet.md#supported-catalog-vs-warm-backends).

## Realtime audio (speech-to-text + text-to-speech)

`lobes init --fleet --audio` adds an **audio overlay** to the fleet: an OpenAI
`/v1/audio/*` facade served by a small **realtime bridge** container, backed by two
GPU sidecars — **Parakeet** (NVIDIA NeMo ASR, `nvidia/parakeet-tdt-0.6b-v2`) for
speech-to-text and **Chatterbox** (Resemble AI, 0.5B, Apache-2.0) for text-to-speech.
No NGC key is required — both are open-weights, pulled from HuggingFace. (Chatterbox
replaced the retired Magpie NIM.)

```bash
lobes init --fleet --audio --apply   # add the audio overlay to the fleet scaffold
lobes fleet up --apply               # build + start STT, TTS, and the realtime bridge
lobes fleet status
```

The gateway fans `/v1/audio/*` out to the bridge, which proxies each request to the
right backend:

```bash
# speech-to-text — multipart upload → {"text": ...}
curl -s http://localhost:8000/v1/audio/transcriptions -F file=@clip.wav

# text-to-speech — text → 24 kHz audio bytes
curl -s http://localhost:8000/v1/audio/speech \
  -d '{"model":"chatterbox","input":"Hello from lobes.","voice":""}' -o speech.wav
```

Chatterbox does zero-shot **voice cloning** — point `DEFAULT_VOICE` (or the request's
`voice`) at a `.wav` path on the sidecar. Verify the whole stack end-to-end with the
TTS → STT round-trip in `scripts/audio-smoke.py`. See
[`docs/realtime-pipeline.md`](docs/realtime-pipeline.md) for the topology and
bring-up, [`docs/parakeet-stt.md`](docs/parakeet-stt.md) +
[`docs/chatterbox-tts.md`](docs/chatterbox-tts.md) for the two backends, or
`lobes explain realtime` for the short version.

## The OpenAI-compatible API surface

Everything lobes serves speaks the OpenAI wire format on **one port** (default
`:8000`), routed by the request's `model` field. Single-model mode serves the
generate endpoints; the fleet adds embeddings, reranking, and (with `--audio`) the
audio endpoints.

| Endpoint | Method | Served by |
|---|---|---|
| `/v1/chat/completions`, `/v1/completions` | POST | generate primary (opt-in fallback) |
| `/v1/embeddings` | POST | Qwen3-Embedding-0.6B gear |
| `/v1/rerank`, `/v1/score` | POST | Qwen3-Reranker-0.6B gear |
| `/v1/audio/transcriptions` | POST | Parakeet STT (audio overlay) |
| `/v1/audio/speech` | POST | Chatterbox TTS (audio overlay) |
| `/v1/models` | GET | the loaded backends (what's hot now) |
| `/v1/models/supported` | GET | the supported catalog (what you can switch to) |
| `/health` | GET | gateway liveness |

See [`docs/openai-api.md`](docs/openai-api.md) for per-endpoint request/response
shapes, the routing rules (name / default / failover / SSE), `curl` examples, and
auth/exposure — or `lobes explain api`.

## lobes is also the deployed agent

`lobes` is one identity, not two: it is the repo/tool that serves the model
*and* the local thinking agent deployed on it. The agent's runtime identity lives
in `AGENTS.md` (the `acp` system prompt) and `culture.yaml` (`suffix: lobes`,
`backend: acp`, `model: vllm-local/sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`) — the same
lobes that runs the engine consumes it over the `acp` `vllm-local` provider.

## Acknowledgements

The serve tuning (the flashinfer attention backend, chunked prefill, async
scheduling, `--max-num-seqs` / `--max-num-batched-tokens`, and the MoE marlin +
MTP speculative-decode flags) and the prompt-heavy / decode-heavy / balanced
workload shapes follow **[shahizat](https://forums.developer.nvidia.com/u/shahizat)**'s
cross-machine NVFP4 benchmark of `Qwen3.6-35B-A3B-NVFP4` on DGX Spark, Jetson
Thor, and Blackwell 6000 Pro:
[*Benchmark Report: Qwen3.6-35B-A3B-NVFP4 on NVIDIA DGX Spark / Jetson Thor / Blackwell 6000 Pro*](https://forums.developer.nvidia.com/t/benchmark-report-qwen3-6-35b-a3b-nvfp4-on-nvidia-dgx-spark-jetson-thor-blackwell-6000-pro/371810)
(NVIDIA Developer Forums, 2026). See [`docs/tuning-profiles.md`](docs/tuning-profiles.md).

Thanks also to **[Mieszko Syty](https://github.com/ms1design)** — AI/ML Engineer at
FutureProofHomes (Warsaw, Poland) and a fellow [Jetson AI Lab](https://www.jetson-ai-lab.com/)
member, the same community shahizat's NVFP4 benchmark comes from — for sharing the
edge-AI serving expertise that this project builds on.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
