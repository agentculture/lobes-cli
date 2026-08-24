#!/usr/bin/env bash
# scripts/spike-preflight.sh — the safety harness for a live cortex spike
# (dspark-speculation-on-the-spark-cortex plan, task t4).
#
# A DSpark spike takes the fleet's PRODUCTION cortex lane down and brings it
# back with a different --speculative-config. Three things can go wrong quietly,
# and this script exists to make each of them loud:
#
#   1. A peer box is proxying `cortex` to this one (or has stopped doing so).
#      Stopping the lane without knowing that turns a local spike into a mesh
#      outage. `preflight` reads every peer's OWN declaration first.
#
#   2. THE SILENT ONE. The fleet compose template records as a MEASURED bug
#      that a brace-containing substitution corrupts compose's interpolation of
#      every LATER brace pair — and PRIMARY_SPECULATIVE_CONFIG was the victim
#      (it lost its closing brace); --hf-overrides must stay LAST and single-
#      quoted so the spark-lobe YaRN JSON survives as ONE argv token. The DSpark
#      config is a longer JSON with a model path, more braces and more spaces.
#      If it gets mangled, vLLM boots HEALTHY and serves with NO speculation —
#      and every tok/s number measured against it is meaningless. So this script
#      never reads the .env line: it reads the argv the CONTAINER actually got,
#      and fails hard on an absent or brace-mangled token.
#
#   3. The lane comes back "up" but not actually serving. `restore` proves
#      recovery with `lobes status` AND one live generate through the gateway.
#
# MUTATION SAFETY (CLAUDE.md convention): `stop` and `restore` are dry-run by
# default and require --apply to touch anything. `preflight` and `check-token`
# are read-only always, and `restore` without --apply only verifies.
#
# Usage:
#   ./scripts/spike-preflight.sh preflight [OPTIONS]
#   ./scripts/spike-preflight.sh stop [--apply] [OPTIONS]
#   ./scripts/spike-preflight.sh restore [--apply] [OPTIONS]
#   ./scripts/spike-preflight.sh check-token '<argv token>'
#
# Modes:
#   preflight     READ-ONLY. Prints local cortex state, every peer's
#                 PRIMARY_PEER_ORIGIN / PRIMARY_PEER_PROXY / PRIMARY_FEASIBLE,
#                 and the container's ACTUAL rendered --speculative-config argv
#                 token. Non-zero if the token is absent or brace-mangled.
#   stop          Runs preflight, prints the operator stop-announcement block,
#                 then stops the cortex lane — only with --apply.
#   restore       Verifies the lane recovered: `lobes status`, the argv proof
#                 again, and one live generate through the gateway. With --apply
#                 it first brings the lane up and waits for health.
#   check-token   READ-ONLY, offline. Validates one argv token given on the
#                 command line and exits 0/1. This is the same validator the
#                 argv proof uses, exposed so it can be tested without docker
#                 (tests/test_spike_preflight.py).
#
# Options:
#   --apply                 Commit the mutation (stop / bring-up). Without it,
#                           those modes print the plan and change nothing.
#   --deploy-dir DIR        Deployment dir (default: $LOBES_DIR, $MODEL_GEAR_DIR,
#                           ~/.lobes, ~/.model-gear)
#   --container NAME        Cortex container (default: model-gear-vllm-primary)
#   --service NAME          Cortex compose service (default: vllm-primary)
#   --peer user@host        Peer to read (repeatable; default: thor@thor and
#                           orin@orin). --peer none reads no peers.
#   --port N                Gateway host port (default: VLLM_PORT in .env, 8000)
#   --base-url URL          Gateway origin (overrides --port)
#   --model NAME            Model/alias for the restore generate (default: cortex)
#   --allow-absent-spec     Treat a MISSING --speculative-config as OK. For the
#                           deliberate `none` arm only — never for a spec arm.
#   --timeout SECS          Health-wait budget for `restore --apply` (default 1500)
#   --ssh-timeout SECS      Per-peer ssh connect timeout (default 5)
#   -h, --help              Show this help and exit
#
# Exit code: 0 iff every check in the selected mode passes.
#
# Run it from a repo checkout on the box that serves the cortex lane (like the
# sibling scripts/live-check.sh and scripts/accept-shape.sh, whose deployment-dir
# resolution and _check idiom this mirrors).

set -euo pipefail

MODE=""
APPLY=0
DEPLOY_DIR=""
CONTAINER="model-gear-vllm-primary"
SERVICE="vllm-primary"
PEERS=()
PEERS_SET=0
PORT=""
BASE_URL=""
GEN_MODEL="cortex"
ALLOW_ABSENT_SPEC=0
TIMEOUT=1500
SSH_TIMEOUT=5
CHECK_TOKEN_ARG=""

_usage() {
  grep '^#' "$0" | sed -n '/^# Usage:/,/^# Exit code:/p' | head -n -1 | sed 's/^# \?//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    preflight|stop|restore) MODE="$1"; shift ;;
    check-token)            MODE="check-token"; CHECK_TOKEN_ARG="${2-}"; shift 2 || shift ;;
    --apply)             APPLY=1; shift ;;
    --deploy-dir)        DEPLOY_DIR="$2"; shift 2 ;;
    --container)         CONTAINER="$2"; shift 2 ;;
    --service)           SERVICE="$2"; shift 2 ;;
    --peer)              PEERS_SET=1; [[ "$2" == "none" ]] || PEERS+=("$2"); shift 2 ;;
    --port)              PORT="$2"; shift 2 ;;
    --base-url)          BASE_URL="$2"; shift 2 ;;
    --model)             GEN_MODEL="$2"; shift 2 ;;
    --allow-absent-spec) ALLOW_ABSENT_SPEC=1; shift ;;
    --timeout)           TIMEOUT="$2"; shift 2 ;;
    --ssh-timeout)       SSH_TIMEOUT="$2"; shift 2 ;;
    -h|--help)           _usage ;;
    *) printf 'error: unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

[[ -n "${MODE}" ]] || { printf 'error: a mode is required (preflight|stop|restore|check-token)\n' >&2; exit 2; }
if [[ ${PEERS_SET} -eq 0 ]]; then PEERS=(thor@thor orin@orin); fi

# check-token is deliberately dependency-free (python3 only) so the offline test
# suite can drive it on a CI box with no docker, no ssh and no fleet.
if [[ "${MODE}" == "check-token" ]]; then
  command -v python3 >/dev/null 2>&1 \
    || { printf 'error: required tool not found: python3\n' >&2; exit 2; }
else
  for dep in docker python3 curl; do
    command -v "${dep}" >/dev/null 2>&1 \
      || { printf 'error: required tool not found: %s\n' "${dep}" >&2; exit 2; }
  done
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------------------
# The argv-token validator. ONE implementation, two entry points: the live
# argv proof pipes docker's own JSON argv array into --tokens-json, and the
# `check-token` mode passes a single token to --single. Keeping it in one place
# is the point — a second copy is a second thing that can be wrong about the
# exact failure this harness exists to catch.
# ---------------------------------------------------------------------------
read -r -d '' VALIDATOR_PY <<'PYEOF' || true
import json
import sys

SPEC_FLAG = "--speculative-config"
# Flags whose value is JSON and therefore brace-bearing. The compose brace bug
# corrupts LATER substitutions, so a mangled --hf-overrides is evidence the same
# breakage is in flight even when the spec token happens to look fine.
JSON_FLAGS = (SPEC_FLAG, "--hf-overrides", "--default-chat-template-kwargs")


def balanced(payload):
    """String-aware brace scan. Returns (ok, reason)."""
    depth = 0
    in_str = False
    esc = False
    for ch in payload:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False, "closing brace before its opening brace"
    if in_str:
        return False, "unterminated JSON string"
    if depth > 0:
        return False, "unbalanced braces: %d closing brace(s) MISSING" % depth
    return True, ""


def validate_payload(payload):
    """Validate the JSON payload of a flag. Returns list of problems (empty = ok)."""
    problems = []
    if payload == "":
        return ["empty payload"]
    if payload[0] in "'\"" or payload[-1] in "'\"":
        # The template single-quotes --hf-overrides on purpose; those quotes are
        # consumed by compose's shell lexer and must NOT survive into argv.
        problems.append("quote character leaked into the argv token (%r ... %r)"
                        % (payload[0], payload[-1]))
    ok, reason = balanced(payload)
    if not ok:
        problems.append(reason)
    try:
        parsed = json.loads(payload)
    except ValueError as exc:
        problems.append("not parseable as JSON: %s" % exc)
    else:
        if not isinstance(parsed, dict):
            problems.append("parsed as %s, expected a JSON object"
                            % type(parsed).__name__)
    return problems


def split_flag(tok):
    """(flag, payload) for a --flag=payload token, else (tok, None)."""
    if tok.startswith("--") and "=" in tok:
        flag, _, payload = tok.partition("=")
        return flag, payload
    return tok, None


def report(label, payload, problems):
    if problems:
        print("  MANGLED  %s" % label)
        print("           token payload: %s" % payload)
        for p in problems:
            print("           -> %s" % p)
    else:
        print("  OK       %s" % label)
        print("           %s" % payload)


def main(argv):
    mode = argv[1]
    if mode == "--single":
        tok = argv[2]
        flag, payload = split_flag(tok)
        if payload is None:
            print("  MANGLED  %s" % tok)
            print("           -> not a --flag=payload token")
            return 1
        if flag not in JSON_FLAGS:
            print("  MANGLED  %s" % flag)
            print("           -> not a JSON-bearing flag this harness validates")
            return 1
        problems = validate_payload(payload)
        report(flag, payload, problems)
        return 1 if problems else 0

    # --tokens-json: docker's own argv array on stdin.
    allow_absent = "--allow-absent-spec" in argv
    tokens = json.load(sys.stdin)
    if not isinstance(tokens, list):
        print("  ERROR    argv is not a JSON array")
        return 1

    failures = 0
    spec_seen = 0
    # Normalise "--flag value" pairs into "--flag=value" so both spellings
    # (compose renders either, depending on the lane) validate identically.
    norm = []
    i = 0
    while i < len(tokens):
        tok = str(tokens[i])
        if tok in JSON_FLAGS and i + 1 < len(tokens):
            norm.append("%s=%s" % (tok, tokens[i + 1]))
            i += 2
            continue
        norm.append(tok)
        i += 1

    for tok in norm:
        flag, payload = split_flag(tok)
        if flag in JSON_FLAGS:
            if flag == SPEC_FLAG:
                spec_seen += 1
            problems = validate_payload(payload if payload is not None else "")
            report(flag, payload, problems)
            if problems:
                failures += 1
        elif payload is None and ("{" in tok or "}" in tok):
            # A bare brace-bearing token nobody declared: exactly what the
            # compose bug produces when it splits a JSON value on a space.
            print("  MANGLED  <orphan brace-bearing argv token>")
            print("           %s" % tok)
            print("           -> a JSON value was split into separate argv tokens")
            failures += 1

    if spec_seen == 0:
        if allow_absent:
            print("  ABSENT   %s (allowed: --allow-absent-spec, the `none` arm)"
                  % SPEC_FLAG)
        else:
            print("  ABSENT   %s" % SPEC_FLAG)
            print("           -> the lane is serving with NO speculative decoding.")
            print("           -> Any tok/s measured against it is NOT a spec result.")
            failures += 1
    elif spec_seen > 1:
        print("  MANGLED  %s appears %d times" % (SPEC_FLAG, spec_seen))
        failures += 1

    return 1 if failures else 0


sys.exit(main(sys.argv))
PYEOF

_validate_single() { python3 -c "${VALIDATOR_PY}" --single "$1"; }

_argv_json() {
  # Prefer .Args (the full resolved argv, entrypoint-relative) but fall back to
  # .Config.Cmd. Both are what the CONTAINER got — never the .env line.
  local j
  j="$(docker inspect "${CONTAINER}" --format '{{json .Args}}' 2>/dev/null || true)"
  if [[ -z "${j}" || "${j}" == "null" || "${j}" == "[]" ]]; then
    j="$(docker inspect "${CONTAINER}" --format '{{json .Config.Cmd}}' 2>/dev/null || true)"
  fi
  printf '%s' "${j}"
}

_argv_proof() { # → 0 iff the rendered spec token is present and well-formed
  printf '\n--- rendered argv proof (source: docker inspect %s, NOT .env) ---\n' "${CONTAINER}"
  local j
  j="$(_argv_json)"
  if [[ -z "${j}" || "${j}" == "null" ]]; then
    printf '  ERROR    container %s not found (or has no argv)\n' "${CONTAINER}"
    return 1
  fi
  local extra=()
  [[ ${ALLOW_ABSENT_SPEC} -eq 1 ]] && extra=(--allow-absent-spec)
  printf '%s' "${j}" | python3 -c "${VALIDATOR_PY}" --tokens-json "${extra[@]}"
}

# ---------------------------------------------------------------------------
# Deployment-dir / port resolution — mirrors scripts/live-check.sh.
# ---------------------------------------------------------------------------
if [[ -z "${DEPLOY_DIR}" ]]; then
  if   [[ -n "${LOBES_DIR:-}" ]];      then DEPLOY_DIR="${LOBES_DIR}"
  elif [[ -n "${MODEL_GEAR_DIR:-}" ]]; then DEPLOY_DIR="${MODEL_GEAR_DIR}"
  elif [[ -d "${HOME}/.lobes" ]];      then DEPLOY_DIR="${HOME}/.lobes"
  elif [[ -d "${HOME}/.model-gear" ]]; then DEPLOY_DIR="${HOME}/.model-gear"
  fi
fi

_env_val() { # _env_val <file> <KEY> — last assignment wins, strip comment/quotes
  grep -E "^[[:space:]]*$2[[:space:]]*=" "$1" 2>/dev/null | tail -n1 \
    | sed -E "s/^[[:space:]]*$2[[:space:]]*=[[:space:]]*//; s/[[:space:]]*#.*$//; s/^\"//; s/\"$//; s/[[:space:]]*$//" || true
}

_env_or_unset() { local v; v="$(_env_val "$1" "$2")"; printf '%s' "${v:-<unset>}"; }

if [[ -z "${BASE_URL}" ]]; then
  if [[ -z "${PORT}" && -n "${DEPLOY_DIR}" ]]; then
    PORT="$(_env_val "${DEPLOY_DIR}/.env" VLLM_PORT)"
  fi
  BASE_URL="http://localhost:${PORT:-8000}"
fi

API_KEY=""
if [[ -n "${DEPLOY_DIR}" && -f "${DEPLOY_DIR}/.env" ]]; then
  API_KEY="$(_env_val "${DEPLOY_DIR}/.env" GATEWAY_API_KEY)"
  [[ -n "${API_KEY}" ]] || API_KEY="$(_env_val "${DEPLOY_DIR}/.env" CULTURE_VLLM_API_KEY)"
fi
AUTH_ARGS=()
[[ -n "${API_KEY}" ]] && AUTH_ARGS=(-H "Authorization: Bearer ${API_KEY}")

STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
THIS_HOST="$(hostname)"

_compose_files() { # the resolved -f chain, from the CLI's single authority
  (cd "${DEPLOY_DIR}" && lobes fleet files --compose-dir "${DEPLOY_DIR}" 2>/dev/null) || true
}

# ---------------------------------------------------------------------------
# Phase 1 — peer state, BEFORE any mutation.
#
# Reads each peer's OWN .env (the operator-typed declaration, per #92 — never
# derived). An unreachable peer is reported as UNREACHABLE and counted; it is
# never silently skipped, because "no output" and "no proxying" must not look
# the same in a transcript.
# ---------------------------------------------------------------------------
PEERS_UNREACHABLE=0
PEERS_PROXYING=""

_peer_block() {
  local peer="$1"
  local ssh_opts=(-o BatchMode=yes -o ConnectTimeout="${SSH_TIMEOUT}"
                  -o StrictHostKeyChecking=accept-new)
  if ! ssh "${ssh_opts[@]}" "${peer}" true >/dev/null 2>&1; then
    printf '  peer %-14s UNREACHABLE (ssh BatchMode, %ss timeout) — state UNKNOWN\n' \
      "${peer}" "${SSH_TIMEOUT}"
    PEERS_UNREACHABLE=$((PEERS_UNREACHABLE + 1))
    return 0
  fi
  local raw
  raw="$(ssh "${ssh_opts[@]}" "${peer}" '
    for d in "${LOBES_DIR:-$HOME/.lobes}" "$HOME/.lobes" "$HOME/.model-gear"; do
      if [ -f "$d/.env" ]; then
        echo "__DIR__=$d"
        grep -hE "^[[:space:]]*(PRIMARY_PEER_ORIGIN|PRIMARY_PEER_PROXY|PRIMARY_FEASIBLE)[[:space:]]*=" "$d/.env" || true
        exit 0
      fi
    done
    echo "__DIR__=<none>"
  ' 2>/dev/null || true)"

  local dir origin proxy feasible
  dir="$(printf '%s\n' "${raw}"  | sed -n 's/^__DIR__=//p' | tail -n1)"
  _pick() {
    printf '%s\n' "${raw}" | grep -E "^[[:space:]]*$1[[:space:]]*=" | tail -n1 \
      | sed -E "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*//; s/[[:space:]]*#.*$//; s/^\"//; s/\"$//; s/[[:space:]]*$//"
  }
  origin="$(_pick PRIMARY_PEER_ORIGIN)"; origin="${origin:-<unset>}"
  proxy="$(_pick PRIMARY_PEER_PROXY)";   proxy="${proxy:-<unset>}"
  feasible="$(_pick PRIMARY_FEASIBLE)";  feasible="${feasible:-<unset>}"

  printf '  peer %-14s reachable   deploy_dir=%s\n' "${peer}" "${dir:-<none>}"
  printf '    PRIMARY_PEER_ORIGIN = %s\n' "${origin}"
  printf '    PRIMARY_PEER_PROXY  = %s\n' "${proxy}"
  printf '    PRIMARY_FEASIBLE    = %s\n' "${feasible}"
  if [[ "${proxy}" == "true" ]]; then
    printf '    !! this peer FORWARDS model=cortex to %s — stopping our lane breaks it\n' "${origin}"
    PEERS_PROXYING="${PEERS_PROXYING}${peer} "
  fi
}

_phase_peers() {
  printf '=== peer cortex state (read BEFORE any mutation) ===\n'
  printf '  read at        : %s\n' "${STAMP}"
  printf '  from           : %s (this box)\n' "${THIS_HOST}"
  if [[ ${#PEERS[@]} -eq 0 ]]; then
    printf '  (no peers requested: --peer none)\n'
  else
    command -v ssh >/dev/null 2>&1 \
      || { printf '  ERROR: ssh not found; peer state UNKNOWN\n'; return 1; }
    local p
    for p in "${PEERS[@]}"; do _peer_block "${p}"; done
  fi
  printf '\n  local cortex declaration (%s/.env):\n' "${DEPLOY_DIR}"
  if [[ -f "${DEPLOY_DIR}/.env" ]]; then
    local k
    for k in PRIMARY_FEASIBLE PRIMARY_PEER_ORIGIN PRIMARY_PEER_PROXY \
             PRIMARY_MODEL PRIMARY_MAX_MODEL_LEN PRIMARY_GPU_MEM_UTIL \
             PRIMARY_SPECULATIVE_CONFIG; do
      printf '    %-24s = %s\n' "${k}" "$(_env_or_unset "${DEPLOY_DIR}/.env" "${k}")"
    done
    printf '    (the PRIMARY_SPECULATIVE_CONFIG line above is NOT the proof —\n'
    printf '     the rendered argv below is. They disagree when compose mangles it.)\n'
  else
    printf '    ERROR: no .env at %s\n' "${DEPLOY_DIR}/.env"
    return 1
  fi
  return 0
}

_phase_container() {
  printf '\n--- cortex container ---\n'
  local line
  line="$(docker ps -a --filter "name=^/${CONTAINER}$" \
            --format '  {{.Names}}  {{.Status}}  image={{.Image}}' 2>/dev/null || true)"
  if [[ -z "${line}" ]]; then
    printf '  (no container named %s)\n' "${CONTAINER}"
    return 1
  fi
  printf '%s\n' "${line}"
  printf '  image digest: %s\n' \
    "$(docker inspect "${CONTAINER}" --format '{{.Image}}' 2>/dev/null || echo '?')"
  return 0
}

_gateway_generate_once() { # → 0 ok, 1 hard fail, 75 transient (429/503, retry)
  local body http out
  # enable_thinking=false mirrors `lobes route`'s terse path: a thinking model
  # otherwise spends the whole token budget in its reasoning trace and returns
  # empty content, which would read as "it did not answer".
  body="$(printf '{"model": "%s", "messages": [{"role": "user", "content": "Reply with the single word: awake."}], "max_tokens": 64, "temperature": 0, "chat_template_kwargs": {"enable_thinking": false}}' "${GEN_MODEL}")"
  out="$(curl -sS -m 180 -w '\n__HTTP__%{http_code}' \
          -H 'Content-Type: application/json' "${AUTH_ARGS[@]}" \
          -d "${body}" "${BASE_URL}/v1/chat/completions" 2>&1 || true)"
  http="$(printf '%s' "${out}" | sed -n 's/.*__HTTP__\([0-9]*\)$/\1/p' | tail -n1)"
  local payload
  payload="$(printf '%s' "${out}" | sed 's/__HTTP__[0-9]*$//')"
  printf '  POST %s/v1/chat/completions  model=%s  -> HTTP %s\n' \
    "${BASE_URL}" "${GEN_MODEL}" "${http:-<none>}"
  if [[ "${http}" == "429" || "${http}" == "503" ]]; then
    # A pressure shed (or an honest warming 503) means the gateway is up and
    # the lane is reachable — it is not a recovery failure by itself. Retried
    # by the caller, and only a run of them fails the check.
    printf '  body: %s\n' "$(printf '%s' "${payload}" | head -c 300)"
    return 75
  fi
  if [[ "${http}" != "200" ]]; then
    printf '  body: %s\n' "$(printf '%s' "${payload}" | head -c 600)"
    return 1
  fi
  printf '%s' "${payload}" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    choice = d["choices"][0]
    msg = choice["message"]
    # vLLM has spelled the trace both ways across builds; accept either, and
    # count it as an answer — a reasoning-only reply still proves the lane
    # generated. Empty on BOTH is the failure.
    text = "".join(str(msg.get(k) or "")
                   for k in ("content", "reasoning_content", "reasoning"))
    finish = choice.get("finish_reason")
except Exception as exc:
    print("  reply: UNPARSEABLE (%s)" % exc)
    sys.exit(1)
text = text.strip()
print("  reply: %r  (finish_reason=%s)" % (text[:160], finish))
sys.exit(0 if text else 1)
'
}

GEN_RETRIES=6
GEN_RETRY_SLEEP=20

_gateway_generate() { # bounded retry around a transient shed → 0 iff it answered
  local attempt rc
  for ((attempt = 1; attempt <= GEN_RETRIES; attempt++)); do
    set +e
    _gateway_generate_once
    rc=$?
    set -e
    [[ ${rc} -eq 0 ]] && return 0
    [[ ${rc} -ne 75 ]] && return 1
    if [[ ${attempt} -lt ${GEN_RETRIES} ]]; then
      printf '  transient (shed/warming) — retry %d/%d in %ss\n' \
        "${attempt}" "$((GEN_RETRIES - 1))" "${GEN_RETRY_SLEEP}"
      sleep "${GEN_RETRY_SLEEP}"
    fi
  done
  printf '  gave up after %d attempts: the gateway never stopped shedding.\n' "${GEN_RETRIES}"
  return 1
}

# ---------------------------------------------------------------------------
# The operator stop-announcement block. Printed BEFORE any stop is issued, in
# both dry-run and --apply, and shaped to paste straight into a mesh channel or
# an evidence transcript.
# ---------------------------------------------------------------------------
_announcement() {
  local proxying_note="none detected"
  [[ -n "${PEERS_PROXYING}" ]] && proxying_note="${PEERS_PROXYING}(these peers forward model=cortex HERE)"
  cat <<EOF

================= OPERATOR STOP ANNOUNCEMENT =================
  ANNOUNCED AT : ${STAMP}
  BOX          : ${THIS_HOST}
  LANE GOING DOWN : ${SERVICE} (${CONTAINER}) — the fleet's \`cortex\` role
  GATEWAY      : ${BASE_URL}
  DEPLOY DIR   : ${DEPLOY_DIR}

  WHY          : DSpark speculation spike — the cortex lane is stopped and
                 re-booted with a different --speculative-config, then measured
                 against its own baseline on this same box.

  IMPACT       : model=cortex / main / hard 404s or fails on this gateway for
                 the duration. Peers proxying cortex here: ${proxying_note}
                 Peers hosting their own cortex are unaffected.

  DURATION     : one model load per arm (the 27B NVFP4 takes minutes, not
                 seconds). Assume the lane is DOWN until an all-clear.

  ALL-CLEAR    : scripts/spike-preflight.sh restore
                 (proves \`lobes status\` + one live generate through the gateway)

  ROLLBACK     : the incumbent config is the .env recorded above. Restoring it
                 and re-running \`restore\` is the abort path.
==============================================================
EOF
}

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
case "${MODE}" in
  check-token)
    [[ -n "${CHECK_TOKEN_ARG}" ]] \
      || { printf 'error: check-token needs an argv token\n' >&2; exit 2; }
    _validate_single "${CHECK_TOKEN_ARG}"
    exit $?
    ;;

  preflight|stop)
    RC=0
    _phase_peers || RC=1
    _phase_container || RC=1
    _argv_proof || RC=1

    printf '\n=== preflight verdict ===\n'
    if [[ ${RC} -eq 0 ]]; then
      printf '  PASS — peer state recorded, and the rendered --speculative-config\n'
      printf '         token is present and well-formed in the container argv.\n'
    else
      printf '  FAIL — see the phases above. If the argv proof failed, DO NOT\n'
      printf '         measure this lane: it is not serving what .env claims.\n'
    fi
    if [[ ${PEERS_UNREACHABLE} -gt 0 ]]; then
      printf '  WARN — %d peer(s) UNREACHABLE; their cortex state is UNKNOWN, not "fine".\n' \
        "${PEERS_UNREACHABLE}"
    fi

    if [[ "${MODE}" == "preflight" ]]; then
      printf '  (read-only mode: nothing was mutated)\n'
      exit "${RC}"
    fi

    # --- stop ---------------------------------------------------------------
    _announcement

    if [[ ${RC} -ne 0 ]]; then
      printf '\nerror: preflight FAILED — refusing to stop the lane on an unproven\n' >&2
      printf 'baseline. Fix the failure above (or pass --allow-absent-spec if the\n' >&2
      printf 'lane is deliberately unspeculated) and re-run.\n' >&2
      exit 1
    fi

    if [[ ${APPLY} -eq 0 ]]; then
      printf '\nDRY RUN — no stop was issued and nothing was mutated.\n'
      printf 'Would run, in %s:\n' "${DEPLOY_DIR}"
      printf '  docker compose %s stop %s\n' \
        "$(_compose_files | tr '\n' ' ')" "${SERVICE}"
      printf 'Re-run with --apply to commit (repo mutation-safety convention).\n'
      exit 0
    fi

    printf '\n--- APPLY: stopping %s ---\n' "${SERVICE}"
    mapfile -t CF < <(_compose_files)
    (cd "${DEPLOY_DIR}" && docker compose "${CF[@]}" stop "${SERVICE}")
    printf 'STOPPED at %s. The lane is DOWN. Announce the all-clear with:\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '  %s/scripts/spike-preflight.sh restore\n' "${REPO_ROOT}"
    exit 0
    ;;

  restore)
    RC=0
    printf '=== restore verification ===\n'
    printf '  at           : %s\n' "${STAMP}"
    printf '  box          : %s\n' "${THIS_HOST}"
    printf '  deploy dir   : %s\n' "${DEPLOY_DIR}"
    printf '  gateway      : %s\n' "${BASE_URL}"

    if [[ ${APPLY} -eq 1 ]]; then
      printf '\n--- APPLY: bringing %s up and waiting for health (<= %ss) ---\n' \
        "${SERVICE}" "${TIMEOUT}"
      mapfile -t CF < <(_compose_files)
      (cd "${DEPLOY_DIR}" && docker compose "${CF[@]}" up -d "${SERVICE}")
      DEADLINE=$(( $(date +%s) + TIMEOUT ))
      while :; do
        H="$(docker inspect "${CONTAINER}" --format '{{.State.Health.Status}}' 2>/dev/null || echo unknown)"
        [[ "${H}" == "healthy" ]] && { printf '  healthy\n'; break; }
        if [[ $(date +%s) -ge ${DEADLINE} ]]; then
          printf '  TIMEOUT — last health status: %s\n' "${H}"
          RC=1
          break
        fi
        sleep 10
      done
    else
      printf '\n  (verify-only: pass --apply to bring the lane up first)\n'
    fi

    printf '\n--- check 1/3: lobes status ---\n'
    if command -v lobes >/dev/null 2>&1; then
      if (cd "${DEPLOY_DIR}" && lobes status --compose-dir "${DEPLOY_DIR}"); then
        printf '  PASS  lobes status\n'
      else
        printf '  FAIL  lobes status\n'; RC=1
      fi
    else
      printf '  FAIL  lobes not on PATH — cannot verify status\n'; RC=1
    fi

    printf '\n--- check 2/3: rendered argv proof (the incumbent config came back intact) ---\n'
    _argv_proof || RC=1

    printf '\n--- check 3/3: one live generate through the gateway ---\n'
    if _gateway_generate; then
      printf '  PASS  live generate\n'
    else
      printf '  FAIL  live generate\n'; RC=1
    fi

    printf '\n=== restore verdict ===\n'
    if [[ ${RC} -eq 0 ]]; then
      printf '  PASS — cortex is back: status healthy, argv intact, and it answered.\n'
    else
      printf '  FAIL — cortex has NOT recovered. Do not announce an all-clear.\n'
    fi
    exit "${RC}"
    ;;
esac
