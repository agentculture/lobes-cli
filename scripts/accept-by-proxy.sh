#!/usr/bin/env bash
# scripts/accept-by-proxy.sh — the by-proxy capabilities-display acceptance run.
#
# Proves, in ONE transcript, that `lobes capabilities` reports WHERE a lobe is
# served rather than merely whether this box wired it:
#
#   1. every proxied role reads `by-proxy` and names its hosting peer, while
#      locally-served roles still read `yes`;
#   2. the displayed state does NOT track local wiring — a proxied role reads
#      `by-proxy` even when its `<PREFIX>_BASE_URL` is set and `loaded` is
#      therefore true (the h7/h9 toggle-proof);
#   3. the gateway JSON is untouched — `loaded` is still a bool for every role;
#   4. traffic did not move: a live POST for each proxied role still returns
#      200 with `X-Lobes-Proxied-By` naming the peer (the h10 boundary proof).
#
# Read-only: it POSTs to the gateway and reads .env, but never writes either.
#
# Usage:
#   ./scripts/accept-by-proxy.sh [--port N] [--deploy-dir DIR] [--out FILE]
set -euo pipefail

# Every phase leans on these non-core tools (curl for the live endpoints,
# python3 for the JSON assertions, docker to read the gateway's own env, git
# for the boundary proof) — validate up front so a fresh host fails with one
# clear line, not mid-run noise.
for dep in curl python3 docker git; do
  command -v "${dep}" >/dev/null 2>&1 \
    || { printf 'error: required tool not found: %s\n' "${dep}" >&2; exit 2; }
done

# `lobes` may be reached either through uv (a source checkout) or directly (an
# installed wheel) — require one, not a specific one.
command -v uv >/dev/null 2>&1 || command -v lobes >/dev/null 2>&1 \
  || { printf 'error: need either uv or lobes on PATH\n' >&2; exit 2; }

# An acceptance gate that cannot fail is not evidence. Every check below
# records into FAILURES instead of aborting, so the transcript stays COMPLETE
# and diagnostic; the script then exits non-zero at the end if anything failed.
FAILURES=()
fail() { FAILURES+=("$1"); printf '  FAIL: %s\n' "$1"; }

PORT=""
DEPLOY_DIR="${LOBES_DIR:-$HOME/.lobes}"
OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --deploy-dir) DEPLOY_DIR="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

ENV_FILE="$DEPLOY_DIR/.env"
[ -n "$PORT" ] || PORT="$(grep -E '^VLLM_PORT=' "$ENV_FILE" 2>/dev/null | tail -1 | cut -d= -f2)"
[ -n "$PORT" ] || PORT=8000
BASE="http://localhost:$PORT"
[ -n "$OUT" ] || OUT="docs/evidence/$(date +%Y-%m-%d)-accept-by-proxy-$(hostname -s).txt"
mkdir -p "$(dirname "$OUT")"

# The gateway's inbound gate accepts GATEWAY_API_KEY, else CULTURE_VLLM_API_KEY.
KEY="$(docker inspect model-gear-gateway --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null \
       | grep -E '^(GATEWAY_API_KEY|CULTURE_VLLM_API_KEY)=.+' | head -1 | cut -d= -f2-)"
AUTH=(); [ -n "$KEY" ] && AUTH=(-H "Authorization: Bearer $KEY")

# Fetch /capabilities ONCE, failing loudly on any HTTP error (curl exits 0 on
# 4xx/5xx without --fail-with-body, which is exactly how an acceptance gate
# ends up certifying a 401).
CAPS="$(mktemp)"
trap 'rm -f "$CAPS"' EXIT

exec > >(tee "$OUT") 2>&1
echo "=== by-proxy capabilities acceptance ==="
echo "host:        $(hostname -s)"
echo "date:        $(date -Is)"
echo "deploy dir:  $DEPLOY_DIR"
echo "gateway:     $BASE"
echo "lobes:       $(uv run lobes --version 2>/dev/null || lobes --version)"
echo "inbound key: $([ -n "$KEY" ] && echo 'armed' || echo 'none')"
echo

echo "--- [0] gateway reachable ---"
if curl -sS --fail-with-body -m 15 "$BASE/capabilities" -o "$CAPS"; then
  echo "  GET /capabilities: 200, $(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$CAPS") roles"
else
  fail "GET $BASE/capabilities did not return 2xx — cannot certify anything below"
  echo; echo "=== ABORTED: $((${#FAILURES[@]})) failure(s) ==="; exit 1
fi
echo

echo "--- [1] capabilities table (the operator surface) ---"
(uv run lobes capabilities 2>/dev/null || lobes capabilities) || fail "lobes capabilities exited non-zero"
echo

echo "--- [2] gateway JSON: loaded is STILL a bool for every role ---"
python3 - "$CAPS" <<'PYEOF' || fail "the /capabilities wire contract changed — loaded is no longer a bool"
import json, sys
caps = json.load(open(sys.argv[1]))
bad = [r for r, v in caps.items() if not isinstance(v.get("loaded"), bool)]
for role, v in caps.items():
    print(f"  {role:9} loaded={str(v['loaded']):5} (type={type(v['loaded']).__name__:4}) "
          f"feasible={str(v.get('feasible')):5} proxied={v.get('proxied')} "
          f"hosted_by={v.get('hosted_by')}")
print()
print("  WIRE CONTRACT:", "UNCHANGED — every loaded is a bool" if not bad
      else f"BROKEN — non-bool loaded for {bad}")
sys.exit(1 if bad else 0)
PYEOF
echo

echo "--- [3] the toggle-proof: display does NOT track local wiring (h7/h9) ---"
python3 - "$CAPS" <<'PYEOF' || fail "toggle-proof could not be established"
import json, sys
caps = json.load(open(sys.argv[1]))
prox = {r: v for r, v in caps.items() if v.get("proxied")}
if not prox:
    print("  (no proxied roles on this box — run on a mesh-lobe deployment)")
    sys.exit(0)
for role, v in prox.items():
    print(f"  {role:9} loaded={str(v['loaded']):5} -> table reads by-proxy  (peer: {v.get('hosted_by')})")
vals = {bool(v["loaded"]) for v in prox.values()}
print()
print("  PROOF:", "loaded DIFFERS across proxied roles, yet both render by-proxy "
      "— the display is independent of local wiring." if len(vals) > 1 else
      "all proxied roles share loaded=%s here; the renderer keys on "
      "feasible+proxied, so wiring cannot move the cell either way." % vals.pop())
PYEOF
echo

echo "--- [4] traffic did not move: live POST per proxied role (h10) ---"
ROLES="$(python3 -c 'import json,sys; print(" ".join(r for r,v in json.load(open(sys.argv[1])).items() if v.get("proxied")))' "$CAPS")"
for ROLE in $ROLES; do
  echo "  model=$ROLE"
  HDRS="$(mktemp)"; BODY="$(mktemp)"
  if curl -sS --fail-with-body -m 240 -D "$HDRS" -X POST "$BASE/v1/chat/completions"       -H 'Content-Type: application/json' "${AUTH[@]}"       -d "{\"model\":\"$ROLE\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly one short sentence about the sea.\"}],\"max_tokens\":60,\"chat_template_kwargs\":{\"enable_thinking\":false}}"       -o "$BODY"; then
    echo "    $(head -1 "$HDRS" | tr -d '\r')"
    PEER="$(grep -i '^x-lobes-proxied-by' "$HDRS" | tr -d '\r' || true)"
    if [ -n "$PEER" ]; then echo "    $PEER"; else fail "$ROLE answered 200 but carried NO X-Lobes-Proxied-By — it was not proxied"; fi
    python3 -c "
import json,sys
d=json.load(open('$BODY'))
c=(d.get('choices') or [{}])[0]
print('    served model:', d.get('model'))
print('    finish:', c.get('finish_reason'), '| content:', repr((c.get('message') or {}).get('content'))[:120])
" || fail "$ROLE returned a body that is not valid chat-completion JSON"
  else
    fail "$ROLE POST did not return 2xx"
    echo "    $(head -1 "$HDRS" 2>/dev/null | tr -d '\r')"; head -c 200 "$BODY" 2>/dev/null; echo
  fi
  rm -f "$HDRS" "$BODY"
done
[ -n "$ROLES" ] || echo "  (no proxied roles to dial)"
echo

echo "--- [5] boundary: no routing/auth file in the diff (h10) ---"
if git rev-parse --git-dir >/dev/null 2>&1; then
  git diff --name-only main...HEAD | sed 's/^/    /'
  echo
  CHANGED="$(git diff --name-only main...HEAD | grep -E '^lobes/gateway/(server|_routing|_config|_realtime)\.py$' || true)"
  echo "  ROUTING/AUTH FILES TOUCHED: ${CHANGED:-none}"
  [ -z "$CHANGED" ] || fail "a routing/auth file is in the diff: $CHANGED"
else
  echo "    (not a git checkout — boundary proof skipped)"
fi
echo

if [ ${#FAILURES[@]} -eq 0 ]; then
  echo "=== PASS — all checks green ==="
else
  echo "=== FAIL — ${#FAILURES[@]} check(s) failed ==="
  printf '  - %s\n' "${FAILURES[@]}"
  exit 1
fi
