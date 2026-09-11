"""Agentic-realistic concurrency test: each stream gets a DISTINCT, LONG prompt
so prefix caching cannot dedupe them. Reports per-stream TTFT/decode, aggregate
decode, and the prefill volume actually served."""
import json, sys, time, threading, urllib.request, random, uuid
URL="http://127.0.0.1:8000/v1/chat/completions"; MODEL="worker"
N=int(sys.argv[1]); APPROX_PROMPT_TOKENS=int(sys.argv[2]) if len(sys.argv)>2 else 4000
SHARED=(sys.argv[3]=="shared") if len(sys.argv)>3 else False
# per-INVOCATION nonce: without it, re-running the sweep would hit the
# prefix cache from the previous width and flatter every later run.
NONCE=uuid.uuid4().hex

WORDS=["alpha","beta","gamma","delta","epsilon","zeta","eta","theta","iota",
       "kappa","lambda","mu","nu","xi","omicron","pi","rho","sigma","tau"]
def make_prompt(seed):
    r=random.Random(NONCE if SHARED else NONCE+str(seed))
    # unique pseudo-source-file per stream: no shared prefix at all
    body="\n".join(
        f"def {r.choice(WORDS)}_{r.randrange(10**6)}(x{r.randrange(100)}, y{r.randrange(100)}):"
        f"  # {' '.join(r.choice(WORDS) for _ in range(12))}\n"
        f"    return x{r.randrange(100)} * {r.randrange(1000)} + y{r.randrange(100)}"
        for _ in range(APPROX_PROMPT_TOKENS//22))
    return ("Here is a source file. Summarise what it does in exactly one "
            f"sentence, then stop.\n\n{body}\n")

res=[]; lock=threading.Lock()
def one(i):
    body={"model":MODEL,"messages":[{"role":"user","content":make_prompt(i)}],
          "max_tokens":200,"temperature":0,"stream":True,
          "stream_options":{"include_usage":True},
          "chat_template_kwargs":{"enable_thinking":False}}
    r=urllib.request.Request(URL,json.dumps(body).encode(),{"Content-Type":"application/json"})
    t0=time.time(); ttft=None; usage=None; last=t0
    with urllib.request.urlopen(r,timeout=None) as x:
        for raw in x:
            if not raw.startswith(b"data: "): continue
            p=raw[6:].strip()
            if p==b"[DONE]": break
            try: ev=json.loads(p)
            except Exception: continue
            if ev.get("usage"): usage=ev["usage"]
            for ch in ev.get("choices") or []:
                c=(ch.get("delta") or {}).get("content")
                if c:
                    if ttft is None: ttft=time.time()-t0
                    last=time.time()
    ct=(usage or {}).get("completion_tokens",0); pt=(usage or {}).get("prompt_tokens",0)
    dec=(ct-1)/((last-t0)-ttft) if ttft and (last-t0)>ttft else 0
    with lock: res.append({"i":i,"ttft_ms":round(ttft*1000,1) if ttft else None,
                           "prompt_tok":pt,"out_tok":ct,"decode_tok_s":round(dec,2)})
T0=time.time()
ths=[threading.Thread(target=one,args=(i,)) for i in range(N)]
[t.start() for t in ths]; [t.join() for t in ths]
WALL=time.time()-T0
out=sum(r["out_tok"] for r in res); pre=sum(r["prompt_tok"] for r in res)
ds=[r["decode_tok_s"] for r in res]
print(json.dumps({"width":N,"prompts":"SHARED" if SHARED else "DISTINCT",
 "prompt_tok_each":res[0]["prompt_tok"] if res else 0,
 "per_stream_decode":sorted(round(d,1) for d in ds),
 "mean_per_stream_decode":round(sum(ds)/len(ds),1),
 "ttft_ms":sorted(r["ttft_ms"] for r in res if r["ttft_ms"]),
 "total_out_tok":out,"total_prompt_tok":pre,"wall_s":round(WALL,2),
 "aggregate_decode_tok_s":round(out/WALL,1),
 "aggregate_prefill_tok_s":round(pre/WALL,1)}))
