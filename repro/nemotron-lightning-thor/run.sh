#!/usr/bin/env bash
# The Jetson AI Lab Thor recipe for Nemotron 3.5 Lightning, verbatim except:
#   - detached (-d, named) instead of -it --rm, so logs survive the wedge
#   - --port 18002 (8000 is occupied on this host)
# Source: https://www.jetson-ai-lab.com/models/nemotron3-5-lightning/#run-on-jetson
set -euo pipefail

docker run -d --pull always \
  --name nemotron35-vllm \
  --runtime=nvidia \
  --network host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/vllm:/root/.cache/vllm" \
  vllm/vllm-openai:v0.27.1 \
  --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --served-model-name nemotron35 \
  --reasoning-parser nemotron_v3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --max-model-len 128000 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.7 \
  --trust-remote-code \
  --max-num-batched-tokens 16384 \
  --enable-prefix-caching \
  --speculative_config.method dspark \
  --speculative_config.model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark \
  --speculative_config.num_speculative_tokens 5 \
  --mamba-backend flashinfer \
  --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 \
  --mamba-cache-mode align \
  --port 18002

echo "waiting for /health on :18002 — on our Thor this never arrives; the"
echo "engine wedges at 'Warming up Mamba2 SSD Triton kernels...' (see README)"
until curl -sf localhost:18002/health >/dev/null; do
  docker logs --tail 1 nemotron35-vllm 2>&1 | tail -1
  sleep 30
done
echo "HEALTHY"
