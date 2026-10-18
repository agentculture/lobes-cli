"""Replay the strict+thinking finish-tool request N times through the Spark gateway (#310 t12/d4)."""
import json, sys, time, urllib.request
S = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
key = next((l.split("=", 1)[1].strip() for l in open("/home/spark/.lobes/.env")
            if l.startswith("GATEWAY_API_KEY=")), "")
body = json.load(open(f"{S}/strict-payload.json"))
for i in range(n):
    req = urllib.request.Request("http://localhost:8001/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=900))
    c = r["choices"][0]; calls = c["message"].get("tool_calls") or []
    print(f"run {i+1}: calls={len(calls)} finish={c['finish_reason']} completion_tokens={r['usage']['completion_tokens']}"
          f" wall={time.time()-t:.1f}s first_args={calls[0]['function']['arguments'][:80] if calls else None}")
