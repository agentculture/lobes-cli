"""strict/thinking controls through the Spark gateway (#310 t12, d5)."""
import json, sys, urllib.request
S = sys.argv[1]
key = next((l.split("=", 1)[1].strip() for l in open("/home/spark/.lobes/.env") if l.startswith("GATEWAY_API_KEY=")), "")
base = json.load(open(f"{S}/strict-payload.json"))
def go(label, b):
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:8001/v1/chat/completions", json.dumps(b).encode(),
        {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}), timeout=900))
    m = r["choices"][0]["message"]; c = m.get("tool_calls") or []; rc = m.get("reasoning_content") or m.get("reasoning") or ""
    print(f"{label}: calls={len(c)} finish={r['choices'][0]['finish_reason']} tokens={r['usage']['completion_tokens']} reasoning_chars={len(rc)} args={c[0]['function']['arguments'][:60] if c else None}")
go("strict, thinking off", dict(base, chat_template_kwargs={"enable_thinking": False}))
nt = json.loads(json.dumps(base)); nt["tools"][0]["function"].pop("strict"); go("non-strict+thinking", nt)
msg = [{"role": "user", "content": "Call the finish tool. The summary argument must be exactly: 'the quick brown fox jumps over the lazy dog'"}]
for strict in (True, False):
    b = json.loads(json.dumps(base)); b["messages"] = msg; f = b["tools"][0]["function"]
    f["parameters"]["properties"]["summary"] = {"type": "string", "maxLength": 3}; f["parameters"]["additionalProperties"] = False
    if not strict: f.pop("strict", None)
    go(f"maxLength control strict={strict}", b)
