#!/usr/bin/env bash
# scripts/validate-tiers.sh — live three-tier-fleet validation harness (t9, #68)
#
# Validates tier alias routing, pressure downgrade (simulated via a lowered
# LOBES_SWAP_DEGRADED_THRESHOLD injected through a docker-compose override),
# and manual override against an already-running fleet.
#
# The fleet MUST already be up (lobes fleet up --apply) with COMPOSE_PROFILES
# including both "middle" and "minor". This script validates — it does NOT deploy,
# switch models, or commit.
#
# Usage:
#   ./scripts/validate-tiers.sh [OPTIONS]
#
# Options:
#   --compose-dir DIR   Deployment dir (default: $LOBES_DIR, $MODEL_GEAR_DIR,
#                       ~/.lobes, or ~/.model-gear — live Spark uses ~/.model-gear)
#   --base-url URL      Gateway OpenAI base URL (default: http://localhost:8001/v1)
#   --out FILE          Write pressure-delta evidence (check F) to this file
#   -h, --help          Show this help and exit
#
# Exit code: 0 if all hard checks (A, C, D, E) pass; 1 if any fail.
# Checks B and F are informational — failures are logged but do not block exit 0.
#
# SAFETY: a trap on EXIT always removes the docker-compose override file and
# restarts the gateway to default thresholds, even on failure or Ctrl-C.
# The fleet is left exactly as found.

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
BASE_URL="http://localhost:8001/v1"
COMPOSE_DIR=""
OUT=""

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
_usage() {
  grep '^#' "$0" | sed -n '/^# Usage:/,/^# [A-Z]/p' | head -n -1 | sed 's/^# \?//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --compose-dir) COMPOSE_DIR="$2"; shift 2 ;;
    --base-url)    BASE_URL="$2";    shift 2 ;;
    --out)         OUT="$2";         shift 2 ;;
    -h|--help)     _usage ;;
    *) printf 'error: unknown option: %s\n' "$1" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Resolve deployment directory
# ---------------------------------------------------------------------------
if [[ -z "${COMPOSE_DIR}" ]]; then
  if   [[ -n "${LOBES_DIR:-}" ]];       then COMPOSE_DIR="${LOBES_DIR}"
  elif [[ -n "${MODEL_GEAR_DIR:-}" ]];  then COMPOSE_DIR="${MODEL_GEAR_DIR}"
  elif [[ -d "${HOME}/.lobes" ]];       then COMPOSE_DIR="${HOME}/.lobes"
  elif [[ -d "${HOME}/.model-gear" ]];  then COMPOSE_DIR="${HOME}/.model-gear"
  else
    printf 'error: cannot resolve deployment dir; pass --compose-dir or set LOBES_DIR\n' >&2
    exit 1
  fi
fi

if [[ ! -f "${COMPOSE_DIR}/docker-compose.yml" ]]; then
  printf 'error: no docker-compose.yml in %s\n' "${COMPOSE_DIR}" >&2
  exit 1
fi

# Strip /v1 so we can hit /health separately
GATEWAY_URL="${BASE_URL%/v1*}"
GATEWAY_URL="${GATEWAY_URL%/v1}"

# ---------------------------------------------------------------------------
# Compose invocation — mirror lobes.runtime._compose._compose_files so a gateway
# recreate keeps the live gateway's full config: the base compose file + the
# audio overlay when scaffolded (it extends the gateway service to route
# /v1/audio/*), plus our temporary threshold override (when present) merged last.
# The override uses a non-auto-discovered name and is ALWAYS passed explicitly,
# so a leftover copy (e.g. after a crash) can never silently merge into a later
# `lobes fleet up`.
# ---------------------------------------------------------------------------
_OVERRIDE_FILE=""     # temporary threshold override written in check D (see below)

_compose_args() {
  local -a _f=(-f "${COMPOSE_DIR}/docker-compose.yml")
  [[ -f "${COMPOSE_DIR}/docker-compose.audio.yml" ]] && _f+=(-f "${COMPOSE_DIR}/docker-compose.audio.yml")
  [[ -n "${_OVERRIDE_FILE}" && -f "${_OVERRIDE_FILE}" ]] && _f+=(-f "${_OVERRIDE_FILE}")
  printf '%s\n' "${_f[@]}"
}

_recreate_gateway() {
  local -a _cf
  mapfile -t _cf < <(_compose_args)
  docker compose --project-directory "${COMPOSE_DIR}" "${_cf[@]}" \
    up -d --no-deps --force-recreate gateway
}

# ---------------------------------------------------------------------------
# Trap: restore gateway on exit (check D/E may lower a threshold)
# ---------------------------------------------------------------------------
_GATEWAY_MODIFIED=false

# shellcheck disable=SC2317  # called indirectly via `trap ... EXIT`
_cleanup() {
  local _rc
  _rc=$?
  printf '\n--- cleanup (rc=%d) ---\n' "${_rc}"
  if [[ -n "${_OVERRIDE_FILE}" && -f "${_OVERRIDE_FILE}" ]]; then
    rm -f "${_OVERRIDE_FILE}"
    printf '  removed compose override: %s\n' "${_OVERRIDE_FILE}"
  fi
  if [[ "${_GATEWAY_MODIFIED}" == true ]]; then
    printf '  recreating gateway with default thresholds...\n'
    _recreate_gateway 2>&1 | tail -3 || true
    printf '  gateway restored to defaults\n'
  fi
}
trap '_cleanup' EXIT

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
for _dep in curl jq nvidia-smi lobes docker; do
  if ! command -v "${_dep}" >/dev/null 2>&1; then
    printf 'error: required tool not found: %s\n' "${_dep}" >&2
    exit 1
  fi
done
if ! docker compose version >/dev/null 2>&1; then
  printf 'error: docker compose (v2) not available\n' >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Pass/fail counters and reporters
# ---------------------------------------------------------------------------
HARD_FAILURES=0
SOFT_NOTES=0

_pass()      { printf 'PASS [%s] %s\n' "$1" "$2"; }
_fail_hard() { printf 'FAIL [%s] %s\n' "$1" "$2"; HARD_FAILURES=$((HARD_FAILURES + 1)); }
_info()      { printf 'INFO [%s] %s\n' "$1" "$2"; SOFT_NOTES=$((SOFT_NOTES + 1)); }

# ---------------------------------------------------------------------------
# Gateway POST helper
# _gateway_post MODEL [EXTRA_HEADER]
# Outputs one line: http_status|body.model|X-Lobes-Tier|X-Lobes-Tier-Reason
# ---------------------------------------------------------------------------
_gateway_post() {
  local _model="$1"
  local _extra="${2:-}"
  local _body _hdrs _status _tier _reason _bmodel

  _body=$(mktemp)
  _hdrs=$(mktemp)

  # shellcheck disable=SC2206
  local _args=(-s -o "${_body}" -D "${_hdrs}" -w '%{http_code}'
    -X POST "${BASE_URL}/chat/completions"
    -H 'Content-Type: application/json'
    -d "{\"model\":\"${_model}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"max_tokens\":4}")
  [[ -n "${_extra}" ]] && _args+=(-H "${_extra}")

  _status=$(curl "${_args[@]}" 2>/dev/null) || _status="000"

  _tier=$(grep -i '^x-lobes-tier:' "${_hdrs}" 2>/dev/null \
          | head -1 | tr -d '\r' | sed 's/^[^:]*:[[:space:]]*//' | tr -d '[:space:]') \
         || _tier=""
  _reason=$(grep -i '^x-lobes-tier-reason:' "${_hdrs}" 2>/dev/null \
            | head -1 | tr -d '\r' | sed 's/^[^:]*:[[:space:]]*//' | tr -d '[:space:]') \
           || _reason=""
  _bmodel=$(jq -r '.model // ""' "${_body}" 2>/dev/null) || _bmodel=""

  rm -f "${_body}" "${_hdrs}"
  printf '%s|%s|%s|%s\n' "${_status}" "${_bmodel}" "${_tier}" "${_reason}"
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
printf '=== lobes three-tier fleet validation (t9 / #68) ===\n'
printf 'compose-dir : %s\n' "${COMPOSE_DIR}"
printf 'base-url    : %s\n' "${BASE_URL}"
printf '\n'

# ===========================================================================
# A. Fleet health
# Five gears must be running: primary (27B), middle (14B), minor (4B),
# embed (0.6B), rerank (0.6B) + the gateway.
# lobes fleet status covers the base four; minor + middle (opt-in profiles)
# are checked via docker inspect directly.
# ===========================================================================
printf '--- A: Fleet health ---\n'

fleet_json=$(lobes fleet status --compose-dir "${COMPOSE_DIR}" --json 2>&1) || fleet_json='{}'

a_failures=0

# Gateway health via lobes fleet status
gw_health=$(printf '%s' "${fleet_json}" | jq -r '.gateway_health // "unknown"' 2>/dev/null \
            || printf 'unknown')
if [[ "${gw_health}" == "ok" ]]; then
  printf '  model-gear-gateway: health=%s\n' "${gw_health}"
else
  _fail_hard 'A' "gateway_health=${gw_health} (expected ok)"
  a_failures=$((a_failures + 1))
fi

# Base fleet containers (primary, embed, rerank) via lobes fleet status
for _cname in "model-gear-vllm-primary" "model-gear-vllm-embed" "model-gear-vllm-rerank"; do
  _cstate=$(printf '%s' "${fleet_json}" \
    | jq -r --arg c "${_cname}" \
      '.containers[] | select(.name == $c) | .state' 2>/dev/null \
    || printf '')
  if [[ -z "${_cstate}" ]]; then
    _fail_hard 'A' "${_cname}: not found in fleet status output"
    a_failures=$((a_failures + 1))
  elif [[ "${_cstate}" != running* ]]; then
    _fail_hard 'A' "${_cname}: state=${_cstate} (expected running*)"
    a_failures=$((a_failures + 1))
  else
    printf '  %s: %s\n' "${_cname}" "${_cstate}"
  fi
done

# Opt-in profile containers (minor + middle) via docker inspect directly
for _cname in "model-gear-vllm-minor" "model-gear-vllm-middle"; do
  _cstate=$(docker inspect \
    -f '{{.State.Status}} ({{.State.Health.Status}})' \
    "${_cname}" 2>/dev/null) || _cstate="not created"
  _cstate="${_cstate:-not created}"
  if [[ "${_cstate}" != running* ]]; then
    _fail_hard 'A' "${_cname}: state=${_cstate} (expected running*; is COMPOSE_PROFILES=minor,middle set?)"
    a_failures=$((a_failures + 1))
  else
    printf '  %s: %s\n' "${_cname}" "${_cstate}"
  fi
done

if [[ "${a_failures}" -eq 0 ]]; then
  _pass 'A' "all five gears running, gateway healthy"
fi
printf '\n'

# ===========================================================================
# B. Memory budget (informational — compare, do not hard-fail)
# Documented: primary 0.45 + middle 0.12 + minor 0.10 + embed 0.06 + rerank 0.06 = 0.79
# Expected usage ~98 GiB / 128 GiB GB10 unified memory.
# ===========================================================================
printf '--- B: Memory budget ---\n'
printf '  documented budget: primary 0.45 + middle 0.12 + minor 0.10 + embed 0.06 + rerank 0.06 = 0.79\n'
printf '  expected usage: ~98 GiB / 128 GiB GB10 unified memory\n'

if _nv_out=$(nvidia-smi --query-gpu=memory.used,memory.total \
               --format=csv,noheader,nounits 2>/dev/null); then
  while IFS=',' read -r _used _total; do
    _used="${_used// /}"; _total="${_total// /}"
    _used_gib=$(( _used / 1024 ))
    _total_gib=$(( _total / 1024 ))
    printf '  GPU memory: %s MiB used / %s MiB total (%d GiB / %d GiB)\n' \
      "${_used}" "${_total}" "${_used_gib}" "${_total_gib}"
  done <<< "${_nv_out}"
  _pass 'B' "nvidia-smi queried — verify ~98 GiB used against 128 GiB total (see above)"
else
  _info 'B' "nvidia-smi unavailable or failed; GPU memory budget not verified"
fi
printf '\n'

# ===========================================================================
# C. Tier alias routing
# POST model=normal → assert X-Lobes-Tier: normal AND body.model ~ Qwen3-14B.
# Smoke-test cheap (4B) and hard (27B) for X-Lobes-Tier match only.
# ===========================================================================
printf '--- C: Tier alias routing ---\n'
c_failures=0

# normal → 14B middle (full assertion: header + body)
printf '  model=normal → X-Lobes-Tier: normal, body.model ~ Qwen3-14B\n'
IFS='|' read -r _st _bm _tier _ < <(_gateway_post "normal")
if [[ "${_st}" != "200" ]]; then
  _fail_hard 'C' "model=normal: HTTP ${_st} (expected 200)"
  c_failures=$((c_failures + 1))
else
  if [[ "${_tier}" != "normal" ]]; then
    _fail_hard 'C' "model=normal: X-Lobes-Tier=${_tier} (expected normal)"
    c_failures=$((c_failures + 1))
  else
    printf '  X-Lobes-Tier: %s (ok)\n' "${_tier}"
  fi
  if [[ "${_bm}" != *"Qwen3-14B"* ]]; then
    _fail_hard 'C' "model=normal: body.model=${_bm} (expected *Qwen3-14B*)"
    c_failures=$((c_failures + 1))
  else
    printf '  body.model:   %s (ok)\n' "${_bm}"
  fi
fi

# cheap → 4B minor (smoke: header only)
printf '  model=cheap → X-Lobes-Tier: cheap\n'
IFS='|' read -r _st _bm _tier _ < <(_gateway_post "cheap")
if [[ "${_st}" != "200" ]]; then
  _fail_hard 'C' "model=cheap: HTTP ${_st}"
  c_failures=$((c_failures + 1))
elif [[ "${_tier}" != "cheap" ]]; then
  _fail_hard 'C' "model=cheap: X-Lobes-Tier=${_tier} (expected cheap)"
  c_failures=$((c_failures + 1))
else
  printf '  cheap: X-Lobes-Tier=%s body.model=%s (ok)\n' "${_tier}" "${_bm}"
fi

# hard → 27B primary (smoke: header only)
printf '  model=hard → X-Lobes-Tier: hard\n'
IFS='|' read -r _st _bm _tier _ < <(_gateway_post "hard")
if [[ "${_st}" != "200" ]]; then
  _fail_hard 'C' "model=hard: HTTP ${_st}"
  c_failures=$((c_failures + 1))
elif [[ "${_tier}" != "hard" ]]; then
  _fail_hard 'C' "model=hard: X-Lobes-Tier=${_tier} (expected hard)"
  c_failures=$((c_failures + 1))
else
  printf '  hard:  X-Lobes-Tier=%s body.model=%s (ok)\n' "${_tier}" "${_bm}"
fi

if [[ "${c_failures}" -eq 0 ]]; then
  _pass 'C' "cheap/normal/hard aliases route to expected tiers"
fi
printf '\n'

# ===========================================================================
# D. Pressure downgrade (simulated — no OOM stressing)
# Technique: read live swap% from lobes status --pressure --json, lower
# LOBES_SWAP_DEGRADED_THRESHOLD below that reading via a temporary
# docker-compose.override.yml, restart only the gateway so it re-imports the
# lowered threshold, then assert model=hard is downgraded to cheap.
# The EXIT trap always removes the override and restores the gateway.
# ===========================================================================
printf '--- D: Pressure downgrade (simulated) ---\n'

# Read live swap%
_p_json=$(lobes status --pressure --json 2>/dev/null) || _p_json='{}'
_real_swap=$(printf '%s' "${_p_json}" | jq -r '.pressure.swap_used_percent // 0' 2>/dev/null \
             || printf '0')
# Floor to integer via awk (handles floats like 22.4 → 22)
_swap_floor=$(awk "BEGIN {printf \"%d\", int(${_real_swap})}" 2>/dev/null) \
             || _swap_floor=0
# Lowered threshold: floor(swap) - 1, minimum 0
if [[ "${_swap_floor}" -gt 0 ]]; then
  _lowered=$(( _swap_floor - 1 ))
else
  _lowered=0
fi

printf '  live swap%%: %s  →  lowering LOBES_SWAP_DEGRADED_THRESHOLD to %d\n' \
  "${_real_swap}" "${_lowered}"

# Write a compose override that injects the lowered threshold into the gateway
# service. It is merged explicitly via -f (see _compose_args) on top of the base
# compose and the audio overlay, so existing gateway env (incl. audio routing) is
# preserved and LOBES_SWAP_DEGRADED_THRESHOLD is appended.
_OVERRIDE_FILE="${COMPOSE_DIR}/docker-compose.validate-tiers.override.yml"
cat > "${_OVERRIDE_FILE}" << OVERRIDE_EOF
# Temporary file written by scripts/validate-tiers.sh (checks D + E).
# Removed automatically by the EXIT trap — do not edit while validate-tiers.sh runs.
# Injects LOBES_SWAP_DEGRADED_THRESHOLD below the live swap% so the pressure-
# downgrade path fires. It is passed explicitly via -f (alongside the base compose
# and the audio overlay when present), so the gateway keeps its full config and a
# leftover copy never auto-merges into a later 'lobes fleet up'. The gateway
# re-imports _pressure_policy at container start, picking up the lowered threshold.
services:
  gateway:
    environment:
      - LOBES_SWAP_DEGRADED_THRESHOLD=${_lowered}
OVERRIDE_EOF

printf '  restarting gateway with lowered threshold...\n'
_GATEWAY_MODIFIED=true
_recreate_gateway 2>&1 | tail -3

# Poll /health for up to 60 s
printf '  waiting for gateway /health'
_health_ok=false
for _i in {1..30}; do
  if curl -sf "${GATEWAY_URL}/health" >/dev/null 2>&1; then
    _health_ok=true
    break
  fi
  printf '.'
  sleep 2
done
printf '\n'

d_failures=0
if [[ "${_health_ok}" != true ]]; then
  _fail_hard 'D' "gateway /health not reachable within 60 s after restart"
  d_failures=$((d_failures + 1))
else
  printf '  model=hard → expect X-Lobes-Tier: cheap, X-Lobes-Tier-Reason: pressure\n'
  IFS='|' read -r _st _bm _tier _reason < <(_gateway_post "hard")

  if [[ "${_st}" != "200" ]]; then
    _fail_hard 'D' "model=hard: HTTP ${_st} (expected 200)"
    d_failures=$((d_failures + 1))
  else
    if [[ "${_tier}" != "cheap" ]]; then
      _fail_hard 'D' "X-Lobes-Tier=${_tier} (expected cheap — downgrade not firing; check that the override file reached the gateway)"
      d_failures=$((d_failures + 1))
    else
      printf '  X-Lobes-Tier:        %s (ok)\n' "${_tier}"
    fi
    if [[ "${_reason}" != "pressure" ]]; then
      _fail_hard 'D' "X-Lobes-Tier-Reason=${_reason} (expected pressure)"
      d_failures=$((d_failures + 1))
    else
      printf '  X-Lobes-Tier-Reason: %s (ok)\n' "${_reason}"
    fi
    if [[ "${d_failures}" -eq 0 ]]; then
      _pass 'D' "model=hard downgraded to cheap (swap=${_real_swap}% > degraded threshold=${_lowered}%, reason=pressure)"
    fi
  fi
fi
printf '\n'

# ===========================================================================
# E. Manual override (override file still active from D)
# X-Lobes-Override: 1 must force the requested tier regardless of pressure.
# ===========================================================================
printf '--- E: Manual override ---\n'
printf '  model=hard + X-Lobes-Override: 1 → X-Lobes-Tier: hard, reason: manual_override\n'
IFS='|' read -r _st _bm _tier _reason < <(_gateway_post "hard" "X-Lobes-Override: 1")

e_failures=0
if [[ "${_st}" != "200" ]]; then
  _fail_hard 'E' "model=hard+override: HTTP ${_st}"
  e_failures=$((e_failures + 1))
else
  if [[ "${_tier}" != "hard" ]]; then
    _fail_hard 'E' "X-Lobes-Tier=${_tier} (expected hard — override did not bypass pressure)"
    e_failures=$((e_failures + 1))
  else
    printf '  X-Lobes-Tier:        %s (ok)\n' "${_tier}"
  fi
  if [[ "${_reason}" != "manual_override" ]]; then
    _fail_hard 'E' "X-Lobes-Tier-Reason=${_reason} (expected manual_override)"
    e_failures=$((e_failures + 1))
  else
    printf '  X-Lobes-Tier-Reason: %s (ok)\n' "${_reason}"
  fi
  if [[ "${e_failures}" -eq 0 ]]; then
    _pass 'E' "model=hard+X-Lobes-Override:1 → hard served, reason=manual_override (pressure bypassed)"
  fi
fi
printf '\n'
# EXIT trap runs after main returns: removes override, recreates gateway at defaults.

# ===========================================================================
# F. Pressure delta evidence (informational)
# Fire small bursts at cheap+normal (4B/14B) vs hard (27B, with override) and
# record swap%/iowait% deltas to show routine work on smaller tiers keeps
# system pressure lower. The lowered threshold from D/E is still active here
# (cleanup runs at EXIT after the script finishes), so we use override for all
# completions to get actual tier dispatch.
# ===========================================================================
printf '--- F: Pressure delta evidence ---\n'
printf '  burst 1: cheap + normal + cheap (4B/14B gears, with override)\n'

_snap_pre_light=$(lobes status --pressure --json 2>/dev/null) || _snap_pre_light='{}'

for _t in cheap normal cheap; do
  _gateway_post "${_t}" "X-Lobes-Override: 1" >/dev/null 2>&1 || true
done

_snap_post_light=$(lobes status --pressure --json 2>/dev/null) || _snap_post_light='{}'

printf '  burst 2: hard + hard + hard (27B gear, with override)\n'
_snap_pre_hard=$(lobes status --pressure --json 2>/dev/null) || _snap_pre_hard='{}'

for _t in hard hard hard; do
  _gateway_post "${_t}" "X-Lobes-Override: 1" >/dev/null 2>&1 || true
done

_snap_post_hard=$(lobes status --pressure --json 2>/dev/null) || _snap_post_hard='{}'

_jq_f() { printf '%s' "$1" | jq -r "$2" 2>/dev/null || printf 'n/a'; }

_sl_swap_pre=$( _jq_f "${_snap_pre_light}"  '.pressure.swap_used_percent // "n/a"')
_sl_swap_post=$(_jq_f "${_snap_post_light}" '.pressure.swap_used_percent // "n/a"')
_sl_io_pre=$(   _jq_f "${_snap_pre_light}"  '.pressure.iowait_percent    // "n/a"')
_sl_io_post=$(  _jq_f "${_snap_post_light}" '.pressure.iowait_percent    // "n/a"')
_sh_swap_pre=$( _jq_f "${_snap_pre_hard}"   '.pressure.swap_used_percent // "n/a"')
_sh_swap_post=$(_jq_f "${_snap_post_hard}"  '.pressure.swap_used_percent // "n/a"')
_sh_io_pre=$(   _jq_f "${_snap_pre_hard}"   '.pressure.iowait_percent    // "n/a"')
_sh_io_post=$(  _jq_f "${_snap_post_hard}"  '.pressure.iowait_percent    // "n/a"')

_evidence=$(cat << EVEOF
F: Pressure delta evidence  ($(date -u +%Y-%m-%dT%H:%M:%SZ))
compose-dir: ${COMPOSE_DIR}

cheap+normal burst (4B/14B gears, 3 completions with X-Lobes-Override):
  swap_before:   ${_sl_swap_pre}%
  swap_after:    ${_sl_swap_post}%
  iowait_before: ${_sl_io_pre}%
  iowait_after:  ${_sl_io_post}%

hard burst (27B gear, 3 completions with X-Lobes-Override):
  swap_before:   ${_sh_swap_pre}%
  swap_after:    ${_sh_swap_post}%
  iowait_before: ${_sh_io_pre}%
  iowait_after:  ${_sh_io_post}%

Interpretation: smaller swap/iowait delta on cheap+normal vs hard shows that
routine workloads on the 4B/14B tiers keep host memory pressure lower than
routing everything to the 27B primary — the core motivation for tiered routing.
EVEOF
)

printf '%s\n' "${_evidence}"

if [[ -n "${OUT}" ]]; then
  printf '%s\n' "${_evidence}" > "${OUT}"
  printf '  evidence written to %s\n' "${OUT}"
fi

_pass 'F' "pressure delta captured (informational — see above)"
printf '\n'

# ===========================================================================
# Summary
# ===========================================================================
printf '=== Summary ===\n'
printf 'Hard checks (A, C, D, E): %d failure(s)\n' "${HARD_FAILURES}"
printf 'Soft notes  (B, F):       %d\n' "${SOFT_NOTES}"
printf '\n'

if [[ "${HARD_FAILURES}" -gt 0 ]]; then
  printf 'RESULT: FAIL — %d hard check(s) did not pass\n' "${HARD_FAILURES}"
  exit 1
else
  printf 'RESULT: PASS — all hard checks passed\n'
  exit 0
fi
