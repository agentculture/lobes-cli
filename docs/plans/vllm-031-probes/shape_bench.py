"""Single-stream decode by content shape through the Spark gateway (#310 t12/t13).

usage: python3 shape_bench.py <model> [runs]   e.g. cortex-spark2 2
Streams 800 tokens per run (thinking off, temperature 0) and prints decode tok/s and TTFT.
"""
import json, sys, time, urllib.request

model = sys.argv[1]; runs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
key = next((l.split("=", 1)[1].strip() for l in open("/home/spark/.lobes/.env") if l.startswith("GATEWAY_API_KEY=")), "")
shapes = {
    "code": "Write a complete Python module implementing a thread-safe LRU cache class with get, put, delete, resize and stats methods, full type hints and docstrings, plus pytest tests for each method.",
    "reasoning": "A train leaves at 14:45 and arrives at 17:10; another leaves at 15:20 and travels 20% faster over the same route. Work out step by step when the second train arrives, then check your answer.",
    "prose": "Write a long, atmospheric short story about an old lighthouse keeper and the last ship he ever guided home.",
}
for name, prompt in shapes.items():
    for i in range(runs):
        body = {"model": model, "max_tokens": 800, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": prompt}], "stream": True, "stream_options": {"include_usage": True}}
        req = urllib.request.Request("http://localhost:8001/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        t0 = time.time(); first = None; n = 0
        for line in urllib.request.urlopen(req, timeout=900):
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("choices") and d["choices"][0]["delta"].get("content") and first is None:
                first = time.time()
            if d.get("usage"):
                n = d["usage"]["completion_tokens"]
        end = time.time()
        print(f"{model} {name:9s} run {i+1}: {n} tok, decode {(n-1)/(end-first):.1f} tok/s (TTFT {first-t0:.2f}s)", flush=True)
