# model-gear

`model-gear` is the tooling that **runs, assesses, and switches** the local,
OpenAI-compatible vLLM model the Culture mesh consumes. The binary is `model` —
`model switch`, `model assess`, `model serve`, and so on.

The served model is what the [`model-gear`](#model-gear-is-also-the-deployed-agent)
agent connects to over the `acp` `vllm-local` provider. The tool and the deployed
agent share one identity: the same model-gear runs the engine and consumes it.

Sibling to [`culture`](https://github.com/agentculture/culture) (the agent mesh),
[`daria`](https://github.com/agentculture/daria) (awareness), and
[`steward`](https://github.com/agentculture/steward) (alignment).

## Install

```bash
uv tool install model-gear
```

## Usage

```bash
model init --apply          # scaffold a deployment dir (default $HOME/.model-gear)
model serve --apply         # start the vLLM server (alias: start)
model switch nvidia/Qwen3-32B-NVFP4 --apply   # switch the served model
model switch nvidia/Qwen3-32B-NVFP4 --purpose decode-heavy --machine spark --apply  # ...in a tuned gear
model status                # current model, container state, /health
model assess                # correctness probes (markdown for a per-model doc)
model benchmark             # decode throughput + prefill latency (shape follows --purpose)
model stop --apply          # stop the server

model overview              # tool snapshot + served model + candidate list
model whoami                # tool, machine, served model, container health
model explain switch        # markdown docs for a topic
model doctor                # diagnose docker / compose / .env / health
```

Every command supports `--json`. **Write verbs (`switch`, `serve`, `stop`,
`init`) are dry-run by default** and require `--apply` to commit — agents call
CLIs in loops, so safe-by-default is mandatory.

## Running the model locally (vLLM)

`model init` scaffolds a deployment directory (default `$HOME/.model-gear`) from the
packaged templates: a `docker-compose.yml` that stands up the vLLM model as an
OpenAI-compatible server on `:8000`, plus a `.env`. Tuned for DGX Spark (GB10
Grace Blackwell, 128 GB unified memory) per
[build.nvidia.com/spark/vllm](https://build.nvidia.com/spark/vllm).

Prerequisites: the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
and `docker login nvcr.io` with an [NGC API key](https://org.ngc.nvidia.com/setup/api-key)
to pull the `nvcr.io/nvidia/vllm` image.

```bash
model init --apply          # writes $HOME/.model-gear/{docker-compose.yml,.env}
# edit $HOME/.model-gear/.env to set HF_TOKEN if the model repo is gated
model serve --apply         # first run downloads ~28 GB of weights (the 27B primary)
model status                # waits/reports until /health is up
```

Verify it is up:

```bash
curl -fsS http://localhost:8000/health
curl -s http://localhost:8000/v1/models   # what's WARM now (the served model), not the catalog
```

Tunables live in the deployment `.env` (`VLLM_MODEL`, `VLLM_GPU_MEM_UTIL`,
`VLLM_MAX_MODEL_LEN`, `HF_CACHE`, …). `VLLM_SERVED_NAME` must match the part
after `vllm-local/` in `culture.yaml` — `model doctor` checks this. `model
switch` rewrites these keys for you.

### Tuning the gear (purpose + machine)

`model switch` resolves the serve config from three layers — the **machine**
profile (`--machine`, default auto-detected: GPU-memory fraction, context,
attention backend), the **workload** profile (`--purpose`, default `balanced`:
the batching knobs and the shape `model benchmark` exercises), and the model's
catalog entry (quantization, tool parser). Explicit `--max-model-len` /
`--gpu-mem-util` flags override the machine defaults.

```bash
model switch sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP --purpose decode-heavy --machine spark --apply
model benchmark --purpose decode-heavy   # shape defaults to the configured VLLM_PURPOSE
model explain tuning                     # the full layering
```

Purposes: `balanced` (≈1K in/1K out), `prompt-heavy` (≈8K in/1K out),
`decode-heavy` (≈1K in/8K out). Machines: `spark` (load-tested), `thor`,
`blackwell`, `generic` (configured). The throughput flags and these shapes follow
shahizat's cross-machine NVFP4 benchmark — see
[`docs/tuning-profiles.md`](docs/tuning-profiles.md).

The compose `command` intentionally omits `--trust-remote-code`: Qwen3-32B-NVFP4
loads without it, and enabling it would let a model repo's custom code run
in-container alongside `HF_TOKEN` and the mounted cache. Add it back only for a
model whose repo ships custom modeling code. If vLLM rejects the `nvidia/`
ModelOpt checkpoint, set `VLLM_MODEL` to the vLLM-native `RedHatAI/Qwen3-32B-NVFP4`
and drop `--quantization` from the compose `command`.

## Expose the API from anywhere (Cloudflare Tunnel)

`model tunnel` publishes the local OpenAI-compatible API at an owner-chosen
hostname through a [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/),
so Culture/AgentCulture agents can call it from anywhere as an ordinary provider
(`base_url` + `api_key`) — no inbound ports, no static IP. The hostname and the
run-token never live in committed config.

> ⚠️ **Gate it first.** A tunnel makes the model reachable from the public
> internet. Set `CULTURE_VLLM_API_KEY` in `$HOME/.model-gear/.env` **before** running
> `model tunnel` — vLLM then requires `Authorization: Bearer $CULTURE_VLLM_API_KEY`
> on every request. Empty leaves the API open; that is only safe for local dev.
> Generate or rotate the key with `python3 scripts/gen-api-key.py` (writes it to
> the gitignored deployment `.env`; `--show` to print, `--force` to rotate), then
> `model serve --apply` to enforce it. You can also set `VLLM_SERVED_NAME` to a
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
cp $HOME/.model-gear/cf-tunnel.env.example $HOME/.model-gear/.cf-tunnel.env
# edit $HOME/.model-gear/.cf-tunnel.env (it is gitignored — never commit it):
#   CULTURE_VLLM_PUBLIC_HOSTNAME=your-host.example
#   CULTURE_CF_TUNNEL_TOKEN_SHUSHU=<shushu-secret-name>

model serve --apply         # serve (with CULTURE_VLLM_API_KEY set in .env)
model tunnel                # DRY RUN: prints the cloudflared command + public URL
model tunnel --apply        # start the tunnel in the background
# ... later:
model tunnel --stop --apply # tear it down
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
deployment; the fleet gateway is not yet auth-aware (planned). See `model explain
tunnel` for the full flow.

## Running the model behind a gateway (fleet)

`model init --fleet` scaffolds a **multi-container** deployment instead of one: the
always-warm Qwen generate primary, two tiny co-resident **embedding** and
**reranker** gears, and a single stdlib **gateway** that fronts them on the host
port the acp `vllm-local` provider already expects. The gateway routes each
request by its `model` field — to the primary, the embedder, or the reranker by
task family (generate / embed / score / rerank) — and defaults an unknown/missing
name to the primary, so existing single-model clients keep working unchanged. The
same front fans `/v1/audio/*` out to the `--audio` overlay, and a warm *generate*
fallback can be wired in later (the gateway adds it, with failover, only when one
is configured).

```bash
model init --fleet --apply        # $HOME/.model-gear/{docker-compose.yml,.env,Dockerfile.gateway}
docker login nvcr.io              # NGC API key for the vLLM image
model fleet up --apply            # builds the gateway image + starts the backend
model fleet status                # container states + gateway /health + /v1/models
```

```bash
curl -s http://localhost:8000/v1/models       # the WARM backend(s) (not the full catalog — see below)
# an unknown/missing model defaults to the primary; route explicitly by name:
curl -s http://localhost:8000/v1/chat/completions -d '{"model":"sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP","messages":[...]}'
```

The fleet runs **one generate backend** by default, so the primary keeps its
load-tested solo headroom (`PRIMARY_GPU_MEM_UTIL=0.6`, full 256K context) on the
128 GB unified memory; the embedding + reranker gears are ~0.6B (util `0.06`
each), so they co-reside without crowding it. To add a warm *generate* fallback,
wire a `vllm-fallback` service + the `FALLBACK_*` env and drop both generate utils
so they sum well under 1.0 (two ~30B NVFP4 models barely co-fit a GB10). `model
switch` drives the single-model deployment (it can also serve an embed/score gear
solo — auto-detected from the catalog, or forced with `--task embed|score`);
change the fleet primary by editing the fleet `.env` and re-running `model fleet
up --apply`. See `model explain fleet` / `model explain gateway` for the routing
semantics, [`docs/qwen3-embedding-0.6b.md`](docs/qwen3-embedding-0.6b.md) +
[`docs/qwen3-reranker-0.6b.md`](docs/qwen3-reranker-0.6b.md) for the gears, and
[`docs/gateway-fleet.md`](docs/gateway-fleet.md) for the full topology.

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
  `PRIMARY_MODEL` / `model switch` when throughput matters more than context/vision.
- [`docs/mistral-small-3.2-24b-nvfp4.md`](docs/mistral-small-3.2-24b-nvfp4.md) —
  the dense **fallback candidate** (`RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4`);
  the default fleet's warm fallback in 0.11.0–0.19.x, since removed (the fleet runs
  one *generate* backend by default — two ~30B NVFP4 models don't co-fit a shared GB10).
  Kept selectable: load-tested 2026-05-30, loads reliably (~15 GiB, ~14.9 tok/s
  decode), text + tool calls (serve with the mistral tokenizer + images disabled).
  Wire it back via the opt-in `FALLBACK_*` fleet config.
- [`docs/qwen3.6-35b-a3b-nvfp4.md`](docs/qwen3.6-35b-a3b-nvfp4.md) — the former
  **MoE fallback** (`mmangkad/Qwen3.6-35B-A3B-NVFP4`), now a candidate. It does
  **not** load reliably on a GB10 shared with other services, and two ~30B models
  do not co-reside there — see [`docs/gateway-fleet.md`](docs/gateway-fleet.md).

The numbers in each doc come from `model switch <model> --apply` then `model
assess` (correctness) and `model benchmark` (throughput). `model overview --list`
lists the catalog (these models) and flags which one is currently served.

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
  model-gear knows how to serve, each tagged `load-tested` (proven on this box) or
  `configured` (declared, not yet proven). It's **static** — defined in
  `model_gear/catalog.py`, shipped in the wheel, unchanged by what's running.
  Read it with `model overview --list` or the gateway's `GET /v1/models/supported`.
- **What's loaded right now?** — the model(s) actually in GPU memory this instant
  (one in single-model mode; in the fleet, the generate primary plus the
  co-resident embedding + reranker gears). The live source is `GET /v1/models`
  (OpenAI-standard); `model fleet status` queries it. `model status` /
  `model whoami` instead report the model the deployment is *configured* to serve
  (from `.env`) plus container health — normally the same model, but it's
  configuration (which can be stale), not a live query.

| Question | CLI | HTTP |
|---|---|---|
| What *can* I run? (catalog) | `model overview --list` | `GET /v1/models/supported` |
| What's *loaded* right now? | `model fleet status` | `GET /v1/models` |
| What's the deployment *set* to serve? | `model status` / `model whoami` | — |

Mnemonic: the catalog is *what's on the menu (and which dishes we've cooked)*;
`/v1/models` is *what's hot now*. See
[`docs/gateway-fleet.md`](docs/gateway-fleet.md#supported-catalog-vs-warm-backends).

## Realtime audio (speech-to-text + text-to-speech)

`model init --fleet --audio` adds an **audio overlay** to the fleet: an OpenAI
`/v1/audio/*` facade served by a small **realtime bridge** container, backed by two
GPU sidecars — **Parakeet** (NVIDIA NeMo ASR, `nvidia/parakeet-tdt-0.6b-v2`) for
speech-to-text and **Chatterbox** (Resemble AI, 0.5B, Apache-2.0) for text-to-speech.
No NGC key is required — both are open-weights, pulled from HuggingFace. (Chatterbox
replaced the retired Magpie NIM.)

```bash
model init --fleet --audio --apply   # add the audio overlay to the fleet scaffold
model fleet up --apply               # build + start STT, TTS, and the realtime bridge
model fleet status
```

The gateway fans `/v1/audio/*` out to the bridge, which proxies each request to the
right backend:

```bash
# speech-to-text — multipart upload → {"text": ...}
curl -s http://localhost:8000/v1/audio/transcriptions -F file=@clip.wav

# text-to-speech — text → 24 kHz audio bytes
curl -s http://localhost:8000/v1/audio/speech \
  -d '{"model":"chatterbox","input":"Hello from model-gear.","voice":""}' -o speech.wav
```

Chatterbox does zero-shot **voice cloning** — point `DEFAULT_VOICE` (or the request's
`voice`) at a `.wav` path on the sidecar. Verify the whole stack end-to-end with the
TTS → STT round-trip in `scripts/audio-smoke.py`. See
[`docs/realtime-pipeline.md`](docs/realtime-pipeline.md) for the topology and
bring-up, [`docs/parakeet-stt.md`](docs/parakeet-stt.md) +
[`docs/chatterbox-tts.md`](docs/chatterbox-tts.md) for the two backends, or
`model explain realtime` for the short version.

## The OpenAI-compatible API surface

Everything model-gear serves speaks the OpenAI wire format on **one port** (default
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
auth/exposure — or `model explain api`.

## model-gear is also the deployed agent

`model-gear` is one identity, not two: it is the repo/tool that serves the model
*and* the local thinking agent deployed on it. The agent's runtime identity lives
in `AGENTS.md` (the `acp` system prompt) and `culture.yaml` (`suffix: model-gear`,
`backend: acp`, `model: vllm-local/sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`) — the same
model-gear that runs the engine consumes it over the `acp` `vllm-local` provider.

## Acknowledgements

The serve tuning (the flashinfer attention backend, chunked prefill, async
scheduling, `--max-num-seqs` / `--max-num-batched-tokens`, and the MoE marlin +
MTP speculative-decode flags) and the prompt-heavy / decode-heavy / balanced
workload shapes follow **[shahizat](https://forums.developer.nvidia.com/u/shahizat)**'s
cross-machine NVFP4 benchmark of `Qwen3.6-35B-A3B-NVFP4` on DGX Spark, Jetson
Thor, and Blackwell 6000 Pro:
[*Benchmark Report: Qwen3.6-35B-A3B-NVFP4 on NVIDIA DGX Spark / Jetson Thor / Blackwell 6000 Pro*](https://forums.developer.nvidia.com/t/benchmark-report-qwen3-6-35b-a3b-nvfp4-on-nvidia-dgx-spark-jetson-thor-blackwell-6000-pro/371810)
(NVIDIA Developer Forums, 2026). See [`docs/tuning-profiles.md`](docs/tuning-profiles.md).

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
