"""Solid-colour vision controls (#310): red/blue/green PNGs at 64 and 224 px.

usage: python3 colour_check.py <base_url> <model> [env_file_with_GATEWAY_API_KEY]
"""
import base64, json, re, struct, sys, urllib.request, zlib

url, model = sys.argv[1].rstrip("/"), sys.argv[2]
key = ""
if len(sys.argv) > 3:
    for line in open(sys.argv[3]):
        m = re.match(r"^GATEWAY_API_KEY=(.*)$", line)
        if m:
            key = m.group(1).strip()


def png(rgb, n):
    raw = b"".join(b"\x00" + bytes(rgb) * n for _ in range(n))
    c = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)  # noqa: E731
    return b"\x89PNG\r\n\x1a\n" + c(b"IHDR", struct.pack(">IIBBBBB", n, n, 8, 2, 0, 0, 0)) + c(b"IDAT", zlib.compress(raw)) + c(b"IEND", b"")


ok = total = 0
for n in (64, 224):
    for name, rgb in (("red", (255, 0, 0)), ("blue", (0, 0, 255)), ("green", (0, 160, 0))):
        img = "data:image/png;base64," + base64.b64encode(png(rgb, n)).decode()
        body = {"model": model, "max_tokens": 300, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img}},
                                                          {"type": "text", "text": "What colour is this image? Answer with one word."}]}]}
        h = {"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})}
        r = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), h), timeout=300))
        a = r["choices"][0]["message"]["content"].strip()
        p = name in a.lower(); ok += p; total += 1
        print(f"{model} {n} px {name} -> {a[:30]} {'PASS' if p else 'FAIL'}")
print(f"{ok}/{total}")
