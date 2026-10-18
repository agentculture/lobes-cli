#!/usr/bin/env bash
# Read-only #310 t17 fleet sweep, run ON one box: vLLM containers' image + vllm.__version__, the .env engine pin
# against the expected template default, vLLM image: lines in the compose override, and the role probes.
# usage: bash fleet_sweep.sh <expected-pin>     (prints values only; never prints .env secrets)
set -u
EXPECTED=$1
D=${LOBES_DIR:-$HOME/.lobes}
export PATH=$HOME/.local/bin:$PATH
echo "box: $(hostname)  lobes: $(lobes --version 2>/dev/null || echo none)  $(date -u +%FT%TZ)"
pin=$(grep -E '^VLLM_NIGHTLY_IMAGE=' "$D/.env" | tail -1 | cut -d= -f2- | awk '{print $1}')
if [ "$pin" = "$EXPECTED" ]; then echo ".env VLLM_NIGHTLY_IMAGE: equals the template default"; else echo ".env VLLM_NIGHTLY_IMAGE: DIFFERS ($pin)"; fi
lines=$(grep -nE '^\s+image:\s*vllm/' "$D"/docker-compose.override.yml 2>/dev/null)
if [ -z "$lines" ]; then echo "override vllm image: lines: none"; else echo "override vllm image: lines: $lines"; fi
echo "containers (the fleet's own, vLLM-based):"
for c in $(docker ps --format '{{.Names}}' | grep -E '^(model-gear-|embed-)' | sort); do
  v=$(docker exec "$c" python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null | tail -1)
  [[ "$v" =~ ^[0-9]+\.[0-9]+ ]] || continue
  id=$(docker inspect -f '{{.Config.Image}}' "$c")
  base=$(docker inspect -f '{{.Image}}' "$c" | cut -c1-19)
  echo "  $c  image=$id  image_id=$base  vllm=$v  $(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}} restarts={{.RestartCount}}' "$c")"
done
echo "probes:"
(cd /tmp && lobes assess --probes 2>&1 | grep -E '^\| [a-z]|Overall')
