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

Tunables live in `.env` (`VLLM_MODEL`, `VLLM_GPU_MEM_UTIL`, `VLLM_MAX_MODEL_LEN`, …).
`VLLM_SERVED_NAME` must match the part after `vllm-local/` in `culture.yaml`.
If vLLM rejects the `nvidia/` ModelOpt checkpoint, set `VLLM_MODEL` to the
vLLM-native `RedHatAI/Qwen3-32B-NVFP4` and drop `--quantization` from the
compose `command`.
