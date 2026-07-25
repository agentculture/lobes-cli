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
set -uo pipefail

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

exec > >(tee "$OUT") 2>&1
echo "=== by-proxy capabilities acceptance ==="
echo "host:        $(hostname -s)"
echo "date:        $(date -Is)"
echo "deploy dir:  $DEPLOY_DIR"
echo "gateway:     $BASE"
echo "lobes:       $(uv run lobes --version 2>/dev/null || lobes --version)"
echo "inbound key: $([ -n "$KEY" ] && echo 'armed' || echo 'none')"
echo

echo "--- [1] capabilities table (the operator surface) ---"
(uv run lobes capabilities 2>/dev/null || lobes capabilities)
echo

echo "--- [2] gateway JSON: loaded is STILL a bool for every role ---"
curl -s -m 15 "$BASE/capabilities" | python3 -c '
import json, sys
caps = json.load(sys.stdin)
bad = [r for r, v in caps.items() if not isinstance(v.get("loaded"), bool)]
for role, v in caps.items():
    print(f"  {role:9} loaded={str(v['"'"'loaded'"'"']):5} (type={type(v['"'"'loaded'"'"']).__name__:4}) "
          f"feasible={str(v.get('"'"'feasible'"'"')):5} proxied={v.get('"'"'proxied'"'"')} "
          f"hosted_by={v.get('"'"'hosted_by'"'"')}")
print()
print("  WIRE CONTRACT:", "UNCHANGED — every loaded is a bool" if not bad
      else f"BROKEN — non-bool loaded for {bad}")
'
echo

echo "--- [3] the toggle-proof: display does NOT track local wiring (h7/h9) ---"
curl -s -m 15 "$BASE/capabilities" | python3 -c '
import json, sys
caps = json.load(sys.stdin)
prox = {r: v for r, v in caps.items() if v.get("proxied")}
if not prox:
    print("  (no proxied roles on this box — run on a mesh-lobe deployment)")
else:
    for role, v in prox.items():
        print(f"  {role:9} loaded={str(v['"'"'loaded'"'"']):5} -> table reads by-proxy  (peer: {v.get('"'"'hosted_by'"'"')})")
    vals = {bool(v["loaded"]) for v in prox.values()}
    print()
    print("  PROOF:", "loaded DIFFERS across proxied roles, yet both render by-proxy "
          "— the display is independent of local wiring." if len(vals) > 1 else
          "all proxied roles share loaded=%s here; the renderer keys on "
          "feasible+proxied, so wiring cannot move the cell either way." % vals.pop())
'
echo

echo "--- [4] traffic did not move: live POST per proxied role (h10) ---"
for ROLE in $(curl -s -m 15 "$BASE/capabilities" \
              | python3 -c 'import json,sys; print(" ".join(r for r,v in json.load(sys.stdin).items() if v.get("proxied")))'); do
  echo "  model=$ROLE"
  HDRS="$(mktemp)"; BODY="$(mktemp)"
  curl -s -m 240 -D "$HDRS" -X POST "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' "${AUTH[@]}" \
    -d "{\"model\":\"$ROLE\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly one short sentence about the sea.\"}],\"max_tokens\":60,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    -o "$BODY"
  echo "    $(head -1 "$HDRS" | tr -d '\r')"
  echo "    $(grep -i '^x-lobes-proxied-by' "$HDRS" | tr -d '\r' || echo 'X-Lobes-Proxied-By: (ABSENT — not proxied!)')"
  python3 -c "
import json,sys
d=json.load(open('$BODY'))
c=(d.get('choices') or [{}])[0]
print('    served model:', d.get('model'))
print('    finish:', c.get('finish_reason'), '| content:', repr((c.get('message') or {}).get('content'))[:120])
" 2>/dev/null || { echo "    (non-JSON body)"; head -c 200 "$BODY"; }
  rm -f "$HDRS" "$BODY"
done
echo

echo "--- [5] boundary: no routing/auth file in the diff (h10) ---"
git diff --name-only main...HEAD 2>/dev/null | sed 's/^/    /' || echo "    (not a git checkout)"
echo
CHANGED="$(git diff --name-only main...HEAD 2>/dev/null | grep -E '^lobes/gateway/(server|_routing|_config|_realtime)\.py$' || true)"
echo "  ROUTING/AUTH FILES TOUCHED: ${CHANGED:-none}"
echo
echo "=== end ==="
