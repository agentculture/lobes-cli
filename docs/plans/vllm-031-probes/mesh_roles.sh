#!/usr/bin/env bash
# Per-role reachability through THIS box's gateway (read-only; key read from .env, never printed).
P=$(grep -E '^VLLM_PORT=' ~/.lobes/.env | cut -d= -f2-); P=${P:-8000}
K=$(grep -E '^GATEWAY_API_KEY=' ~/.lobes/.env | cut -d= -f2-)
H=(-H "Authorization: Bearer $K" -H 'Content-Type: application/json')
out=""
for m in cortex worker hand; do
  c=$(curl -s -o /dev/null -w '%{http_code}' "${H[@]}" localhost:$P/v1/chat/completions -d "{\"model\":\"$m\",\"max_tokens\":4,\"chat_template_kwargs\":{\"enable_thinking\":false},\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}")
  out="$out $m=$c"
done
c=$(curl -s -o /dev/null -w '%{http_code}' "${H[@]}" localhost:$P/v1/embeddings -d '{"model":"embedder","input":"hello"}'); out="$out embedder=$c"
c=$(curl -s -o /dev/null -w '%{http_code}' "${H[@]}" localhost:$P/v1/rerank -d '{"model":"reranker","query":"q","documents":["a","b"]}'); out="$out reranker=$c"
echo "$(hostname):$out"
