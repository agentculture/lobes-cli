#!/usr/bin/env bash
# scripts/live-check.sh — the single-trigger local "advertised implies reachable"
# gate (plan of the same name, task t9).
#
# Runs `tests/test_live_capabilities.py` against a RUNNING deployment and returns
# a pass/fail exit code. One command, unattended, no prompts, no per-role
# babysitting: it resolves the deployment port the way the CLI does, arms the
# gate via LOBES_SMOKE_BASE_URL, runs pytest, and prints a short human summary.
#
# This is a LOCAL pre-PR gate, not a CI job — CI has no GPU and no fleet. Run it
# from a repo checkout, against the box that serves the model, before opening a
# PR. It is strictly READ-ONLY: it dials the gateway and never restarts,
# rebuilds, switches, or otherwise mutates any container.
#
# Usage:
#   ./scripts/live-check.sh [OPTIONS] [-- <extra pytest args>]
#
# Options:
#   --port N            Gateway host port (default: VLLM_PORT in .env, else 8000)
#   --compose-dir DIR   Deployment dir (default: $LOBES_DIR, $MODEL_GEAR_DIR,
#                       ~/.lobes, or ~/.model-gear)
#   --base-url URL      Full gateway origin to test (overrides port resolution,
#                       e.g. http://localhost:8001). Default: http://localhost:<port>
#   -h, --help          Show this help and exit
#
# Exit code: 0 iff every check passes; non-zero on any 404 on an advertised path,
# any unreachable endpoint, a CLI/gateway disagreement, or a version skew. A 429
# (pressure shed) or a 503+Retry-After (honest warming/dead-owner) is treated as
# reachable and does NOT fail the gate.
#
# Why a shell script and not a Makefile target: the job is imperative — resolve a
# deployment dir with a fallback chain, parse a port out of .env, compose a base
# URL, export an env var, run pytest, and translate the result into a one-line
# human verdict. That is script logic, not a dependency graph; there is no
# Makefile in this repo, and `scripts/` already holds the sibling live harness
# `validate-tiers.sh`, whose deployment-dir resolution this mirrors.

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults / argument parsing
# ---------------------------------------------------------------------------
PORT=""
COMPOSE_DIR=""
BASE_URL=""
PYTEST_EXTRA=()

_usage() {
  grep '^#' "$0" | sed -n '/^# Usage:/,/^# Exit code:/p' | head -n -1 | sed 's/^# \?//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)        PORT="$2";        shift 2 ;;
    --compose-dir) COMPOSE_DIR="$2"; shift 2 ;;
    --base-url)    BASE_URL="$2";    shift 2 ;;
    -h|--help)     _usage ;;
    --)            shift; PYTEST_EXTRA=("$@"); break ;;
    *) printf 'error: unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Resolve the base URL the way the CLI resolves its port:
#   --base-url wins outright; else --port; else VLLM_PORT in the deployment's
#   .env; else 8000. Deployment dir precedence mirrors lobes.runtime._compose
#   (and scripts/validate-tiers.sh): --compose-dir → $LOBES_DIR →
#   $MODEL_GEAR_DIR → ~/.lobes → ~/.model-gear.
# ---------------------------------------------------------------------------
_read_env_port() {
  # Echo the VLLM_PORT value from an .env file (last assignment wins), stripping
  # an inline "# comment", surrounding quotes, and whitespace. Empty if absent.
  local envf="$1"
  [[ -f "${envf}" ]] || return 0
  grep -E '^[[:space:]]*VLLM_PORT[[:space:]]*=' "${envf}" \
    | tail -n1 \
    | sed -E 's/^[[:space:]]*VLLM_PORT[[:space:]]*=[[:space:]]*//; s/[[:space:]]*#.*$//; s/^"//; s/"$//; s/[[:space:]]*$//'
}

if [[ -z "${BASE_URL}" ]]; then
  if [[ -z "${PORT}" ]]; then
    if [[ -z "${COMPOSE_DIR}" ]]; then
      if   [[ -n "${LOBES_DIR:-}" ]];      then COMPOSE_DIR="${LOBES_DIR}"
      elif [[ -n "${MODEL_GEAR_DIR:-}" ]]; then COMPOSE_DIR="${MODEL_GEAR_DIR}"
      elif [[ -d "${HOME}/.lobes" ]];      then COMPOSE_DIR="${HOME}/.lobes"
      elif [[ -d "${HOME}/.model-gear" ]]; then COMPOSE_DIR="${HOME}/.model-gear"
      fi
    fi
    if [[ -n "${COMPOSE_DIR}" ]]; then
      PORT="$(_read_env_port "${COMPOSE_DIR}/.env")"
    fi
    PORT="${PORT:-8000}"
  fi
  BASE_URL="http://localhost:${PORT}"
fi

# ---------------------------------------------------------------------------
# Pick the pytest runner: prefer uv (this repo's documented workflow), fall back
# to a plain `python -m pytest` so the gate still runs outside a uv checkout.
# ---------------------------------------------------------------------------
if command -v uv >/dev/null 2>&1; then
  RUNNER=(uv run pytest)
  PYBIN=(uv run python)
else
  RUNNER=(python -m pytest)
  PYBIN=(python)
fi

CLI_VERSION="$("${PYBIN[@]}" -c 'import lobes; print(lobes.__version__)' 2>/dev/null || echo '?')"

printf '=== lobes live capabilities gate ===\n'
printf '  deployment dir : %s\n' "${COMPOSE_DIR:-<not resolved; using default base-url>}"
printf '  gateway origin : %s\n' "${BASE_URL}"
printf '  cli version    : %s\n' "${CLI_VERSION}"
printf '  invariant      : advertised implies reachable (404/unreachable/skew fail; 429/503+Retry-After pass)\n\n'

# ---------------------------------------------------------------------------
# Arm the gate and run. LOBES_SMOKE_BASE_URL set ⇒ the module fails (never skips)
# on any fault. We do NOT pass -x: the operator wants the full picture — every
# role and model reported in one run.
# ---------------------------------------------------------------------------
export LOBES_SMOKE_BASE_URL="${BASE_URL}"

set +e
"${RUNNER[@]}" tests/test_live_capabilities.py -v "${PYTEST_EXTRA[@]}"
RC=$?
set -e

printf '\n=== result ===\n'
if [[ ${RC} -eq 0 ]]; then
  printf '  PASS — everything the deployment advertises is reachable, and %s matches.\n' "${BASE_URL}"
else
  printf '  FAIL (exit %d) — an advertised capability is NOT reachable, or the deployed\n' "${RC}"
  printf '  gateway is version-skewed from this CLI. See the per-check detail above:\n'
  printf '  each fault names the role/model and the issue (#91/#92/#95/#96/#99) it maps to.\n'
fi
exit "${RC}"
