"""Per-modality negative controls for the gemma2-embed lane (#310 t10), run on the Orin.

Same design as docs/evidence/2026-10-07-accept-orin-embed.txt section 2: a text query must rank
the matching synthetic item first of >= 3. Image: solid colours. Audio: 440 Hz tone vs noise vs
near-silence. Code: three snippets. Video is not covered here (needs ffmpeg-built clips).
Usage: python3 eg2_controls.py <gateway_base_url>   (GATEWAY_API_KEY read from ~/.lobes/.env)
"""

import base64
import io
import json
import math
import random
import re
import struct
import sys
import urllib.request
import wave
import zlib
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
KEY = ""
for line in (Path.home() / ".lobes/.env").read_text().splitlines():
    m = re.match(r"^GATEWAY_API_KEY=(.*)$", line)
    if m:
        KEY = m.group(1).strip()


def post(body):
    req = urllib.request.Request(
        BASE + "/v1/embeddings",
        json.dumps(body).encode(),
        {"Content-Type": "application/json", **({"Authorization": f"Bearer {KEY}"} if KEY else {})},
    )
    return json.load(urllib.request.urlopen(req, timeout=300))["data"][0]["embedding"]


def text(q):
    return post({"model": "gemma2-embed", "input": q})


def media(part):
    return post({"model": "gemma2-embed", "messages": [{"role": "user", "content": [part]}]})


def png(rgb, n=64):
    raw = b"".join(b"\x00" + bytes(rgb) * n for _ in range(n))
    c = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)  # noqa: E731
    return b"\x89PNG\r\n\x1a\n" + c(b"IHDR", struct.pack(">IIBBBBB", n, n, 8, 2, 0, 0, 0)) + c(b"IDAT", zlib.compress(raw)) + c(b"IEND", b"")


def wav(kind, secs=2.0, rate=16000):
    rnd = random.Random(7)
    frames = []
    for i in range(int(secs * rate)):
        if kind == "tone":
            v = 0.6 * math.sin(2 * math.pi * 440 * i / rate)
        elif kind == "noise":
            v = rnd.uniform(-0.6, 0.6)
        else:  # near-silence
            v = rnd.uniform(-0.002, 0.002)
        frames.append(struct.pack("<h", int(v * 32767)))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(frames))
    return buf.getvalue()


def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def probe(name, query, items, expect):
    q = text(query)
    sims = {k: round(cos(q, v), 4) for k, v in items.items()}
    top = max(sims, key=sims.get)
    ok = top == expect
    print(json.dumps({"probe": name, "query": query, "sims": sims, "top": top, "pass": ok}))
    return ok


b64 = lambda b: base64.b64encode(b).decode()  # noqa: E731
results = [
    probe(
        "text->image",
        "a solid red square",
        {c: media({"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64(png(rgb))}})
         for c, rgb in (("red", (255, 0, 0)), ("blue", (0, 0, 255)), ("green", (0, 160, 0)))},
        "red",
    ),
    probe(
        "text->audio",
        "a pure musical tone, a steady beep",
        {k: media({"type": "input_audio", "input_audio": {"data": b64(wav(k)), "format": "wav"}})
         for k in ("tone", "noise", "silence")},
        "tone",
    ),
    probe(
        "text->code",
        "merge two sorted lists into one sorted list",
        {
            "merge": text("def merge(a, b):\n    out = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n    return out + a[i:] + b[j:]"),
            "weather": text("def get_weather(city):\n    return requests.get(f'https://api.weather/{city}').json()['temp']"),
            "http": text("def serve(port):\n    HTTPServer(('', port), Handler).serve_forever()"),
        },
        "merge",
    ),
]
print("ALL_PASS" if all(results) else "SOME_FAILED")
