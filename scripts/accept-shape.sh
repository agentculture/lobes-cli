#!/usr/bin/env bash
# scripts/accept-shape.sh — the brain-shapes acceptance run (#113 plan, t6/t7).
#
# One command, unattended, fail-not-skip: move the current deployment aside,
# scaffold the requested shape from a clean dir (dry-run shown first, then
# --apply), boot the fleet, prove dropped-lobe honesty and per-role correctness
# LIVE, run the advertised-implies-reachable gate, measure the reclaimed heavy-
# lobe budget, and leave a full transcript. `--restore` puts the previous
# deployment back.
#
# Usage:
#   ./scripts/accept-shape.sh <machine-as-brain|spark-lobe|thor-lobe|orin-small> [OPTIONS]
#   ./scripts/accept-shape.sh --restore [--deploy-dir DIR]
#
# Options:
#   --deploy-dir DIR   Deployment dir (default: $LOBES_DIR, else ~/.lobes)
#   --audio            Scaffold the audio overlay too (stt/tts lanes)
#   --dev-version V    Dev lane: override MODEL_GEAR_VERSION with a published
#                      TestPyPI .devN build of THIS branch (see Dockerfile.gateway)
#   --dev-index URL    Dev lane extra pip index (default with --dev-version:
#                      https://test.pypi.org/simple/)
#   --port N           Host port for the gateway (sed into VLLM_PORT after the
#                      scaffold — for boxes where the default 8000 is taken)
#   --env KEY=VAL      Append an operator env var to the scaffolded .env
#                      (repeatable — e.g. the documented #100 pressure-threshold
#                      workaround on boxes with phantom swap/iowait readings)
#   --timeout SECS     Health-wait budget for the fleet boot (default 1500)
#   --restore          Restore the newest <deploy-dir>.pre-accept-* backup, boot
#                      it, and exit
#   -h, --help         Show this help and exit
#
# Transcript: ~/lobes-accept-<shape>-<UTC-stamp>.log (path printed at exit).
# Exit code: 0 iff every phase passes. Any honesty violation, failed probe,
# unreachable advertised endpoint, or version skew is a non-zero exit.
#
# Run it from a repo checkout on the box that serves the fleet (like the
# sibling scripts/live-check.sh, whose deployment-dir resolution this mirrors).

set -euo pipefail

SHAPE=""
DEPLOY_DIR=""
AUDIO=0
DEV_VERSION=""
DEV_INDEX=""
PORT_OVERRIDE=""
ENV_OVERRIDES=()
TIMEOUT=1500
RESTORE=0

_usage() {
  grep '^#' "$0" | sed -n '/^# Usage:/,/^# Run it/p' | sed 's/^# \?//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    machine-as-brain|spark-lobe|thor-lobe|orin-small) SHAPE="$1"; shift ;;
    --deploy-dir)  DEPLOY_DIR="$2"; shift 2 ;;
    --audio)       AUDIO=1; shift ;;
    --dev-version) DEV_VERSION="$2"; shift 2 ;;
    --dev-index)   DEV_INDEX="$2"; shift 2 ;;
    --port)        PORT_OVERRIDE="$2"; shift 2 ;;
    --env)         ENV_OVERRIDES+=("$2"); shift 2 ;;
    --timeout)     TIMEOUT="$2"; shift 2 ;;
    --restore)     RESTORE=1; shift ;;
    -h|--help)     _usage ;;
    *) printf 'error: unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${DEPLOY_DIR}" ]]; then
  DEPLOY_DIR="${LOBES_DIR:-${HOME}/.lobes}"
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

FAILURES=0
_check() { # _check <label> <cmd...>  — run, count failure, keep going
  local label="$1"; shift
  if "$@"; then
    printf '  PASS  %s\n' "${label}"
  else
    printf '  FAIL  %s\n' "${label}"
    FAILURES=$((FAILURES + 1))
  fi
}

_env_val() { # _env_val <file> <KEY> — last assignment wins, strip comment/quotes
  grep -E "^[[:space:]]*$2[[:space:]]*=" "$1" 2>/dev/null | tail -n1 \
    | sed -E "s/^[[:space:]]*$2[[:space:]]*=[[:space:]]*//; s/[[:space:]]*#.*$//; s/^\"//; s/\"$//; s/[[:space:]]*$//" || true
}

_port() { local p; p="$(_env_val "${DEPLOY_DIR}/.env" VLLM_PORT)"; echo "${p:-8000}"; }

_compose_down() { # best-effort down of whatever compose lives in $1
  local dir="$1"
  [[ -f "${dir}/docker-compose.yml" ]] || return 0
  local files=(-f "${dir}/docker-compose.yml")
  [[ -f "${dir}/docker-compose.audio.yml" ]] && files+=(-f "${dir}/docker-compose.audio.yml")
  (cd "${dir}" && docker compose "${files[@]}" down --remove-orphans) || true
}

_wait_healthy() { # _wait_healthy <port> <timeout-secs>
  local port="$1" budget="$2" start now
  start="$(date +%s)"
  printf 'waiting for gateway /health + all hosted roles ready (budget %ss)...\n' "${budget}"
  while :; do
    now="$(date +%s)"
    if (( now - start > budget )); then
      printf 'TIMEOUT waiting for healthy fleet\n'
      return 1
    fi
    if curl -fsS "http://localhost:${port}/health" >/dev/null 2>&1; then
      if curl -fsS "http://localhost:${port}/capabilities" 2>/dev/null | python3 -c '
import json, sys
caps = json.load(sys.stdin)
roles = caps.get("roles", caps)
# Core lanes only: the audio pair is owned by phase 3b (and only when the
# --audio overlay is scaffolded) — a no-audio deployment honestly reports
# stt/tts ready=false forever, which must not stall the core-fleet wait.
bad = [n for n, r in roles.items()
       if n not in ("stt", "tts")
       and isinstance(r, dict) and r.get("feasible", True) and not r.get("ready", False)]
sys.exit(1 if bad else 0)
' 2>/dev/null; then
        printf 'fleet healthy after %ss\n' "$((now - start))"
        return 0
      fi
    fi
    sleep 10
  done
}

# ---------------------------------------------------------------------------
# --restore mode
# ---------------------------------------------------------------------------
if [[ "${RESTORE}" -eq 1 ]]; then
  BACKUP="$(ls -d "${DEPLOY_DIR}".pre-accept-* 2>/dev/null | sort | tail -n1 || true)"
  [[ -n "${BACKUP}" ]] || { printf 'error: no %s.pre-accept-* backup found\n' "${DEPLOY_DIR}" >&2; exit 2; }
  printf '=== restore: %s -> %s ===\n' "${BACKUP}" "${DEPLOY_DIR}"
  _compose_down "${DEPLOY_DIR}"
  rm -rf "${DEPLOY_DIR}.accepted-${STAMP}"
  [[ -d "${DEPLOY_DIR}" ]] && mv "${DEPLOY_DIR}" "${DEPLOY_DIR}.accepted-${STAMP}"
  mv "${BACKUP}" "${DEPLOY_DIR}"
  files=(-f "${DEPLOY_DIR}/docker-compose.yml")
  [[ -f "${DEPLOY_DIR}/docker-compose.audio.yml" ]] && files+=(-f "${DEPLOY_DIR}/docker-compose.audio.yml")
  (cd "${DEPLOY_DIR}" && docker compose "${files[@]}" up -d)
  _wait_healthy "$(_port)" "${TIMEOUT}"
  printf '=== restore complete (shape run kept at %s.accepted-%s) ===\n' "${DEPLOY_DIR}" "${STAMP}"
  exit 0
fi

[[ -n "${SHAPE}" ]] || { printf 'error: shape required (machine-as-brain|spark-lobe|thor-lobe|orin-small)\n' >&2; exit 2; }
if [[ -n "${DEV_VERSION}" && -z "${DEV_INDEX}" ]]; then DEV_INDEX="https://test.pypi.org/simple/"; fi

TRANSCRIPT="${HOME}/lobes-accept-${SHAPE}-${STAMP}.log"
exec > >(tee "${TRANSCRIPT}") 2>&1

printf '=== lobes brain-shapes acceptance: %s ===\n' "${SHAPE}"
printf '  utc          : %s\n' "$(date -u +%FT%TZ)"
printf '  host         : %s\n' "$(hostname)"
printf '  repo rev     : %s\n' "$(git rev-parse --short HEAD 2>/dev/null || echo '?')"
printf '  cli version  : %s\n' "$(uv run python -c 'from importlib.metadata import version; print(version("lobes-cli"))')"
printf '  deploy dir   : %s\n' "${DEPLOY_DIR}"
printf '  dev lane     : %s\n' "${DEV_VERSION:-<off — released version>}"
printf '  transcript   : %s\n\n' "${TRANSCRIPT}"

# Shape-specific expectations. DROPPED_ROLES are role names for the
# capabilities feasible:false checks; DROPPED_ALIASES are every generate alias
# that must 404. orin-small (mesh-brain end-state t2, issue #112) drops BOTH
# heavies and serves the opt-in 4B minor instead — running it on a non-Orin
# card exercises the shape contract only and validates nothing about Orin
# (the #108 rule stands until a physical Orin boots it).
MINOR_CURL=0
case "${SHAPE}" in
  spark-lobe)
    DROPPED_ROLES=(senses); DROPPED_ALIASES=(senses multimodal normal)
    HEAVY_PREFIX="PRIMARY"; HEAVY_ALIAS="main"
    PROBES=(cortex embedder reranker); SENSES_CURL=0 ;;
  thor-lobe)
    DROPPED_ROLES=(cortex); DROPPED_ALIASES=(cortex main hard)
    HEAVY_PREFIX="MULTIMODAL"; HEAVY_ALIAS="multimodal"
    PROBES=(embedder reranker); SENSES_CURL=1 ;;
  orin-small)
    DROPPED_ROLES=(cortex senses)
    DROPPED_ALIASES=(cortex main hard senses multimodal normal)
    HEAVY_PREFIX="VLLM_MINOR"; HEAVY_ALIAS="minor"
    PROBES=(embedder reranker); SENSES_CURL=0; MINOR_CURL=1 ;;
  machine-as-brain)
    DROPPED_ROLES=(); DROPPED_ALIASES=()
    HEAVY_PREFIX="PRIMARY"; HEAVY_ALIAS="main"
    PROBES=(cortex embedder reranker); SENSES_CURL=1 ;;
esac

# ---------------------------------------------------------------------------
# Phase 1 — take the current deployment down and move it aside
# ---------------------------------------------------------------------------
printf '=== phase 1: backup current deployment ===\n'
if [[ -d "${DEPLOY_DIR}" ]]; then
  _compose_down "${DEPLOY_DIR}"
  mv "${DEPLOY_DIR}" "${DEPLOY_DIR}.pre-accept-${STAMP}"
  printf 'moved %s -> %s.pre-accept-%s\n\n' "${DEPLOY_DIR}" "${DEPLOY_DIR}" "${STAMP}"
else
  printf 'no existing deployment at %s — clean start\n\n' "${DEPLOY_DIR}"
fi
# Fixed container_name entries would collide if a stray project still holds
# them (e.g. a deployment started from a pre-rename dir) — sweep leftovers.
LEFT="$(docker ps -aq --filter name=model-gear- 2>/dev/null || true)"
if [[ -n "${LEFT}" ]]; then
  printf 'sweeping leftover model-gear-* containers from a stray project\n\n'
  # shellcheck disable=SC2086
  docker rm -f ${LEFT} >/dev/null
fi

# ---------------------------------------------------------------------------
# Phase 2 — scaffold the shape: dry-run diff first, then --apply
# ---------------------------------------------------------------------------
printf '=== phase 2: lobes init --shape %s (dry-run, then --apply) ===\n' "${SHAPE}"
INIT_FLAGS=(--shape "${SHAPE}")
[[ "${AUDIO}" -eq 1 ]] && INIT_FLAGS+=(--audio)
uv run lobes init "${INIT_FLAGS[@]}" "${DEPLOY_DIR}"
uv run lobes init "${INIT_FLAGS[@]}" --apply "${DEPLOY_DIR}"
printf '\n'

if [[ -n "${PORT_OVERRIDE}" ]]; then
  printf '=== phase 2a: host port override — VLLM_PORT=%s ===\n\n' "${PORT_OVERRIDE}"
  sed -i -E "s|^VLLM_PORT=.*$|VLLM_PORT=${PORT_OVERRIDE}|" "${DEPLOY_DIR}/.env"
fi

if [[ "${#ENV_OVERRIDES[@]}" -gt 0 ]]; then
  printf '=== phase 2a2: operator env overrides ===\n'
  for kv in "${ENV_OVERRIDES[@]}"; do
    printf '  %s\n' "${kv}"
    printf '%s\n' "${kv}" >> "${DEPLOY_DIR}/.env"
  done
  printf '\n'
fi

# Dev lane: pin the gateway image to the published .devN of this branch.
if [[ -n "${DEV_VERSION}" ]]; then
  printf '=== phase 2b: dev lane — MODEL_GEAR_VERSION=%s via %s ===\n' "${DEV_VERSION}" "${DEV_INDEX}"
  sed -i -E "s|^MODEL_GEAR_VERSION=.*$|MODEL_GEAR_VERSION=${DEV_VERSION}|" "${DEPLOY_DIR}/.env"
  printf 'GATEWAY_PIP_EXTRA_INDEX_URL=%s\n' "${DEV_INDEX}" >> "${DEPLOY_DIR}/.env"
  printf 'pinned (documented operator vars — see Dockerfile.gateway)\n\n'
fi

# ---------------------------------------------------------------------------
# Phase 3 — boot and wait
# ---------------------------------------------------------------------------
printf '=== phase 3: lobes fleet up --apply ===\n'
uv run lobes fleet up --apply --compose-dir "${DEPLOY_DIR}"
PORT="$(_port)"
_wait_healthy "${PORT}" "${TIMEOUT}"
printf '\n'

BASE="http://localhost:${PORT}"

# ---------------------------------------------------------------------------
# Phase 3b — audio warm-up (first STT/TTS calls trigger model loads; checking
# the lanes before they are warm reports the warm-up window, not the shape)
# ---------------------------------------------------------------------------
if [[ "${AUDIO}" -eq 1 ]]; then
  printf '=== phase 3b: audio warm-up (TTS + STT first-call load) ===\n'
  AWORK="$(mktemp -d /tmp/lobes-accept-audio.XXXXXX)"
  python3 - "${AWORK}/tiny.wav" <<'PY'
import struct, sys, wave
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(struct.pack("<3200h", *([0] * 3200)))  # 0.2 s silence
PY
  AUDIO_START="$(date +%s)"
  for lane in tts stt; do
    while :; do
      if [[ "${lane}" == tts ]]; then
        CODE="$(curl -s -o /dev/null -w '%{http_code}' "${BASE}/v1/audio/speech" \
          -H 'Content-Type: application/json' -d '{"model":"tts-1","input":"ok"}' || echo 000)"
      else
        CODE="$(curl -s -o /dev/null -w '%{http_code}' "${BASE}/v1/audio/transcriptions" \
          -F "file=@${AWORK}/tiny.wav;type=audio/wav" -F model=stt-1 || echo 000)"
      fi
      # Warm means the relay reached a live backend: 2xx (or a parameter-level
      # 4xx) — a 5xx means the sidecar is still loading its model.
      if [[ "${CODE}" =~ ^[23-4] ]]; then
        printf '  %s warm (HTTP %s, %ss)\n' "${lane}" "${CODE}" "$(( $(date +%s) - AUDIO_START ))"
        break
      fi
      if (( $(date +%s) - AUDIO_START > TIMEOUT )); then
        printf '  %s NOT warm within budget (last HTTP %s)\n' "${lane}" "${CODE}"
        FAILURES=$((FAILURES + 1))
        break
      fi
      sleep 10
    done
  done
  printf '\n'
fi

# ---------------------------------------------------------------------------
# Phase 4 — dropped-lobe honesty (#92 invariant, per dropped alias)
# ---------------------------------------------------------------------------
printf '=== phase 4: dropped-lobe honesty ===\n'
WORK="$(mktemp -d /tmp/lobes-accept.XXXXXX)"
uv run lobes capabilities --compose-dir "${DEPLOY_DIR}" --json > "${WORK}/caps-cli.json"
curl -fsS "${BASE}/capabilities" > "${WORK}/caps-gw.json"
curl -fsS "${BASE}/v1/models" > "${WORK}/models.json"

_role_infeasible() { # _role_infeasible <caps-file> <role>
  python3 - "$1" "$2" <<'PY'
import json, sys
caps = json.load(open(sys.argv[1]))
roles = caps.get("roles", caps)
r = roles.get(sys.argv[2])
sys.exit(0 if isinstance(r, dict) and r.get("feasible") is False else 1)
PY
}

if [[ "${#DROPPED_ALIASES[@]}" -gt 0 ]]; then
  for DROPPED_ROLE in "${DROPPED_ROLES[@]}"; do
    _check "CLI capabilities: ${DROPPED_ROLE} feasible:false (flagged, not hidden)" \
      _role_infeasible "${WORK}/caps-cli.json" "${DROPPED_ROLE}"
    _check "gateway /capabilities agrees: ${DROPPED_ROLE} feasible:false" \
      _role_infeasible "${WORK}/caps-gw.json" "${DROPPED_ROLE}"
  done
  _check "/v1/models omits the dropped role's aliases" \
    python3 - "${WORK}/models.json" "${DROPPED_ALIASES[@]}" <<'PY'
import json, sys
ids = {m["id"] for m in json.load(open(sys.argv[1]))["data"]}
sys.exit(1 if ids & set(sys.argv[2:]) else 0)
PY
  for alias in "${DROPPED_ALIASES[@]}"; do
    _check "POST model=${alias} -> 404 role_infeasible (never rerouted)" bash -c "
      resp=\$(curl -s -w '\n%{http_code}' '${BASE}/v1/chat/completions' \
        -H 'Content-Type: application/json' \
        -d '{\"model\":\"${alias}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}')
      code=\${resp##*$'\n'}; body=\${resp%$'\n'*}
      [[ \"\${code}\" == 404 ]] && grep -q role_infeasible <<<\"\${body}\"
    "
  done
else
  printf '  (machine-as-brain: no dropped roles — honesty phase is the shape contract itself)\n'
fi
printf '\n'

# ---------------------------------------------------------------------------
# Phase 4b — honest referral (mesh-brain t3, issue #112). Opt-in: only checked
# for a dropped role whose <PREFIX>_PEER_ORIGIN is declared in the deployed
# .env (pass it with --env). Annotation only — the origin is named on the two
# honesty surfaces; the gateway never dials it (so a dead peer cannot fail
# this phase, by design).
# ---------------------------------------------------------------------------
printf '=== phase 4b: honest referral (opt-in peer origins) ===\n'
_role_prefix() { # role name -> env prefix
  case "$1" in
    cortex) echo PRIMARY ;; senses) echo MULTIMODAL ;;
    embedder) echo EMBED ;; reranker) echo RERANK ;; *) echo "" ;;
  esac
}
_hosted_by() { # _hosted_by <caps-file> <role> <origin> — hosted_by matches
  python3 - "$1" "$2" "$3" <<'PY'
import json, sys
caps = json.load(open(sys.argv[1]))
roles = caps.get("roles", caps)
r = roles.get(sys.argv[2])
sys.exit(0 if isinstance(r, dict) and r.get("hosted_by") == sys.argv[3] else 1)
PY
}
REFERRALS=0
for role in "${DROPPED_ROLES[@]:-}"; do
  [[ -n "${role}" ]] || continue
  prefix="$(_role_prefix "${role}")"
  [[ -n "${prefix}" ]] || continue
  origin="$(_env_val "${DEPLOY_DIR}/.env" "${prefix}_PEER_ORIGIN")"
  [[ -n "${origin}" ]] || continue
  REFERRALS=$((REFERRALS + 1))
  _check "gateway /capabilities: ${role} hosted_by ${origin}" \
    _hosted_by "${WORK}/caps-gw.json" "${role}" "${origin}"
  _check "CLI capabilities: ${role} hosted_by ${origin}" \
    _hosted_by "${WORK}/caps-cli.json" "${role}" "${origin}"
  alias_for_role="${role}"
  _check "POST model=${alias_for_role} -> 404 body carries the referral" bash -c "
    curl -s '${BASE}/v1/chat/completions' -H 'Content-Type: application/json' \
      -d '{\"model\":\"${alias_for_role}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}' \
      | python3 -c 'import json,sys; b=json.load(sys.stdin); e=b[\"error\"]; sys.exit(0 if e.get(\"code\")==\"role_infeasible\" and e.get(\"hosted_by\")==\"${origin}\" else 1)'
  "
done
if [[ "${REFERRALS}" -eq 0 ]]; then
  printf '  (no <PREFIX>_PEER_ORIGIN declared for a dropped role — referral is opt-in, skipped)\n'
fi
printf '\n'

# ---------------------------------------------------------------------------
# Phase 5 — per-role correctness probes (healthy-but-wrong FAILS)
# ---------------------------------------------------------------------------
printf '=== phase 5: correctness probes ===\n'
for role in "${PROBES[@]}"; do
  _check "probe: ${role}" uv run lobes assess --probes --role "${role}" --compose-dir "${DEPLOY_DIR}"
done
if [[ "${SENSES_CURL}" -eq 1 ]]; then
  _check "probe: senses text known-answer (model=multimodal)" bash -c "
    curl -fsS '${BASE}/v1/chat/completions' -H 'Content-Type: application/json' \
      -d '{\"model\":\"multimodal\",\"messages\":[{\"role\":\"user\",\"content\":\"What color is a cloudless daytime sky? Answer with one word.\"}],\"max_tokens\":16}' \
      | python3 -c 'import json,sys; body=json.load(sys.stdin); sys.exit(0 if \"blue\" in body[\"choices\"][0][\"message\"][\"content\"].lower() else 1)'
  "
fi
printf '\n'

# ---------------------------------------------------------------------------
# Phase 6 — advertised implies reachable + version skew (live-check gate)
# ---------------------------------------------------------------------------
printf '=== phase 6: scripts/live-check.sh ===\n'
_check "advertised-implies-reachable gate" ./scripts/live-check.sh --compose-dir "${DEPLOY_DIR}"
printf '\n'

# ---------------------------------------------------------------------------
# Phase 7 — measured budget (numbers, not assertions)
# ---------------------------------------------------------------------------
printf '=== phase 7: measured heavy-lobe budget ===\n'
printf '  %s_GPU_MEM_UTIL   = %s\n' "${HEAVY_PREFIX}" "$(_env_val "${DEPLOY_DIR}/.env" "${HEAVY_PREFIX}_GPU_MEM_UTIL")"
printf '  %s_MAX_MODEL_LEN  = %s\n' "${HEAVY_PREFIX}" "$(_env_val "${DEPLOY_DIR}/.env" "${HEAVY_PREFIX}_MAX_MODEL_LEN")"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null \
  | sed 's/^/  gpu memory     = /' || true
printf '  decode sample  : '
curl -fsS "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"${HEAVY_ALIAS}\",\"messages\":[{\"role\":\"user\",\"content\":\"Count from 1 to 30, comma separated.\"}],\"max_tokens\":64}" \
  -o /tmp/accept-decode.json -w 'HTTP %{http_code} in %{time_total}s' || true
python3 -c "
import json
body = json.load(open('/tmp/accept-decode.json'))
u = body.get('usage', {})
print(f\"  ({u.get('completion_tokens', '?')} completion tokens)\")
" 2>/dev/null || printf '\n'
printf '\n'

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
printf '=== verdict ===\n'
if [[ "${FAILURES}" -eq 0 ]]; then
  printf '  PASS — shape %s validated live on %s. Transcript: %s\n' "${SHAPE}" "$(hostname)" "${TRANSCRIPT}"
  printf '  restore the previous deployment with: ./scripts/accept-shape.sh --restore\n'
  exit 0
else
  printf '  FAIL — %d check(s) failed. Transcript: %s\n' "${FAILURES}" "${TRANSCRIPT}"
  exit 1
fi
