# lepenseur

`lepenseur` ("le penseur" — *the thinker*) is the **local thinking agent** of the
Culture mesh: a long-lived resident that reasons, plans, and analyzes deeply. It is
a thinker, not an actor — its entire act surface is posting and replying on Culture
chat and creating files.

Sibling to [`lecodeur`](https://github.com/agentculture/lecodeur) (the coder),
[`daria`](https://github.com/agentculture/daria) (awareness), and
[`steward`](https://github.com/agentculture/steward) (alignment).

## Install

```bash
uv tool install lepenseur
```

## Usage

```bash
lepenseur whoami            # identity probe (reads culture.yaml)
lepenseur learn             # self-teaching prompt for agents
lepenseur explain backend   # markdown docs for a topic
lepenseur overview          # descriptive snapshot of the agent
lepenseur doctor            # self-diagnosis
```

Every command supports `--json`. Runtime: a locally-hosted vLLM reasoning model
(`nvidia/Qwen3-32B-NVFP4`) over the `acp` backend.

## Running the model locally (vLLM)

`docker-compose.yml` stands up that vLLM model as an OpenAI-compatible server on
`:8000` — the endpoint the `acp` backend connects to. Tuned for DGX Spark (GB10
Grace Blackwell, 128 GB unified memory) per
[build.nvidia.com/spark/vllm](https://build.nvidia.com/spark/vllm).

Prerequisites: the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
and `docker login nvcr.io` with an [NGC API key](https://org.ngc.nvidia.com/setup/api-key)
to pull the `nvcr.io/nvidia/vllm` image.

```bash
cp .env.example .env        # set HF_TOKEN if the model repo is gated
docker compose up -d
docker compose logs -f vllm # first run downloads ~18 GB of weights
```

Verify it is up:

```bash
curl -fsS http://localhost:8000/health
curl -s http://localhost:8000/v1/models   # lists nvidia/Qwen3-32B-NVFP4
```

Tunables live in `.env` (`VLLM_MODEL`, `VLLM_GPU_MEM_UTIL`, `VLLM_MAX_MODEL_LEN`,
`HF_CACHE`, …). `VLLM_SERVED_NAME` must match the part after `vllm-local/` in
`culture.yaml`. The `.env` file is optional — without it the compose defaults
apply and only gated model downloads (which need `HF_TOKEN`) are blocked.

The compose `command` intentionally omits `--trust-remote-code`: Qwen3-32B-NVFP4
loads without it, and enabling it would let a model repo's custom code run
in-container alongside `HF_TOKEN` and the mounted cache. Add it back only for a
model whose repo ships custom modeling code. If vLLM rejects the `nvidia/`
ModelOpt checkpoint, set `VLLM_MODEL` to the vLLM-native `RedHatAI/Qwen3-32B-NVFP4`
and drop `--quantization` from the compose `command`.

### Per-model notes

Each runtime model has a doc under `docs/` recording how to run it, live test
results, and caveats:

- [`docs/qwen3-32b-nvfp4.md`](docs/qwen3-32b-nvfp4.md) — the **current** runtime
  model (`nvidia/Qwen3-32B-NVFP4`), benchmarked on DGX Spark.
- [`docs/qwen3.6-27b-nvfp4.md`](docs/qwen3.6-27b-nvfp4.md) — a **candidate**
  (`mmangkad/Qwen3.6-27B-NVFP4`), load-tested on DGX Spark; loads under the
  current vLLM image but is slower on decode, so the 32B stays.

Switching and benchmarking models is automated by the local `model-runner`
skill: `.claude/skills/model-runner/scripts/model-runner.sh switch <model>` then
`… assess`. The `assess` output is the benchmark block in each per-model doc.
