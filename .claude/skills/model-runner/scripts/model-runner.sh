#!/usr/bin/env bash
# model-runner — switch and assess/benchmark lepenseur's local vLLM runtime model.
#
# Drives the repo's own docker-compose.yml + .env. No paths outside this repo.
# Subcommands:
#   switch MODEL [--port P] [--max-model-len N] [--served-name N] [--gpu-mem-util F]
#   assess [--port P] [--model NAME] [--decode-tokens N]
#   status
#   down
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null || (cd "$HERE/../../../.." && pwd))"
ENV_FILE="$ROOT/.env"
CONTAINER="lepenseur-vllm"

die() { echo "model-runner: $*" >&2; exit 1; }

_set_env() {  # _set_env KEY VALUE — update KEY=VALUE in .env, or append if absent
  local key="$1" val="$2"
  [ -f "$ENV_FILE" ] || die ".env not found at $ENV_FILE — run 'cp .env.example .env' first"
  python3 - "$ENV_FILE" "$key" "$val" <<'PY'
import sys
path, key, val = sys.argv[1], sys.argv[2], sys.argv[3]
lines = open(path).read().splitlines()
seen = False
out = []
for ln in lines:
    if ln.startswith(key + "="):
        out.append(f"{key}={val}"); seen = True
    else:
        out.append(ln)
if not seen:
    out.append(f"{key}={val}")
open(path, "w").write("\n".join(out) + "\n")
PY
}

_get_env() {  # _get_env KEY [DEFAULT]
  local v; v="$(grep -E "^$1=" "$ENV_FILE" 2>/dev/null | head -1 | cut -d= -f2- || true)"
  echo "${v:-${2:-}}"
}

_wait_health() {  # _wait_health PORT
  local port="$1" deadline; deadline=$(( $(date +%s) + 2700 ))
  echo ">> waiting for /health on :$port (first run downloads weights; up to 45 min)"
  while :; do
    [ "$(date +%s)" -ge "$deadline" ] && die "timeout waiting for health on :$port"
    if curl -fsS -m 3 "http://localhost:$port/health" >/dev/null 2>&1; then
      echo ">> healthy on :$port"; return 0
    fi
    local state; state="$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo missing)"
    if [ "$state" != "running" ]; then
      docker logs "$CONTAINER" 2>&1 | tail -30 >&2
      die "container is '$state' — load failed (logs above)"
    fi
    sleep 15
  done
}

cmd_switch() {
  [ $# -ge 1 ] || die "usage: switch MODEL [--port P] [--max-model-len N] [--served-name N] [--gpu-mem-util F]"
  local model="$1"; shift
  local port="8001" maxlen="32768" served="" gpu="0.6"
  while [ $# -gt 0 ]; do
    case "$1" in
      --port) port="$2"; shift 2;;
      --max-model-len) maxlen="$2"; shift 2;;
      --served-name) served="$2"; shift 2;;
      --gpu-mem-util) gpu="$2"; shift 2;;
      *) die "unknown flag: $1";;
    esac
  done
  [ -n "$served" ] || served="$model"
  echo ">> switching to $model (port=$port max_model_len=$maxlen served-name=$served gpu-mem-util=$gpu)"
  _set_env VLLM_MODEL "$model"
  _set_env VLLM_SERVED_NAME "$served"
  _set_env VLLM_PORT "$port"
  _set_env VLLM_MAX_MODEL_LEN "$maxlen"
  _set_env VLLM_GPU_MEM_UTIL "$gpu"
  ( cd "$ROOT" && docker compose down && docker compose up -d )
  _wait_health "$port"
  echo ">> done. assess with: $(basename "$0") assess --port $port"
}

cmd_assess() {
  local port; port="$(_get_env VLLM_PORT 8000)"
  local model; model="$(_get_env VLLM_SERVED_NAME)"
  local decode="512"
  while [ $# -gt 0 ]; do
    case "$1" in
      --port) port="$2"; shift 2;;
      --model) model="$2"; shift 2;;
      --decode-tokens) decode="$2"; shift 2;;
      *) die "unknown flag: $1";;
    esac
  done
  # Host-side facts first
  local image; image="$(docker inspect "$CONTAINER" --format '{{.Config.Image}}' 2>/dev/null || echo '?')"
  local gpu_mem; gpu_mem="$(nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader 2>/dev/null \
      | grep -i 'EngineCore\|vllm' | head -1 | awk -F, '{print $2}' | xargs || echo '?')"
  echo "### Host-side"
  echo "- Image: \`$image\`  ·  GPU memory (EngineCore): ${gpu_mem:-?}"
  echo
  python3 "$HERE/_assess.py" --url "http://localhost:$port" ${model:+--model "$model"} --decode-tokens "$decode"
}

cmd_status() {
  local port; port="$(_get_env VLLM_PORT 8000)"
  echo "model:  $(_get_env VLLM_MODEL '(unset)')"
  echo "served: $(_get_env VLLM_SERVED_NAME '(unset)')  port: $port"
  echo "state:  $(docker inspect -f '{{.State.Status}} ({{.State.Health.Status}})' "$CONTAINER" 2>/dev/null || echo 'not created')"
  curl -fsS -m 3 "http://localhost:$port/health" >/dev/null 2>&1 && echo "health: OK (:$port)" || echo "health: not responding (:$port)"
}

cmd_down() { ( cd "$ROOT" && docker compose down ); }

case "${1:-}" in
  switch) shift; cmd_switch "$@";;
  assess) shift; cmd_assess "$@";;
  status) shift; cmd_status "$@";;
  down)   shift; cmd_down "$@";;
  *) die "usage: $(basename "$0") {switch|assess|status|down} [args]  (see SKILL.md)";;
esac
