"""Streaming measurement: TTFT from the first content delta, decode tok/s from
the streamed deltas after it. No timeouts."""

import json
import sys
import time
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8100/v1/chat/completions"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "nvidia/Qwen3.6-35B-A3B-NVFP4"


def stream(prompt, max_tokens, thinking=False, label=""):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    req = urllib.request.Request(
        URL, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    t0 = time.time()
    ttft = None
    n = 0
    last = t0
    text = []
    usage = None
    error_found = None
    with urllib.request.urlopen(req, timeout=None) as r:  # NO timeout
        for raw in r:
            if not raw.startswith(b"data: "):
                continue
            payload = raw[6:].strip()
            if payload == b"[DONE]":
                break
            try:
                ev = json.loads(payload)
            except Exception:
                continue
            # Detect error events from the gateway and surface them.
            if ev.get("error"):
                error_found = ev["error"]
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                c = d.get("content") or d.get("reasoning_content")
                if c:
                    if ttft is None:
                        ttft = time.time() - t0
                    n += 1
                    last = time.time()
                    text.append(c)
    total = last - t0
    decode_s = (last - t0 - ttft) if ttft is not None else 0.0
    # If usage is present, use completion_tokens; otherwise mark as None
    # rather than falling back to delta count (which is not token-accurate).
    completion_tokens = (usage or {}).get("completion_tokens")
    if completion_tokens is None:
        completion_tokens = None  # no token count — keep it None so the caller knows
    prompt_tokens = (usage or {}).get("prompt_tokens")
    if error_found:
        return {
            "label": label,
            "error": error_found,
            "ttft_ms": None,
            "deltas": n,
            "completion_tokens": None,
            "prompt_tokens": None,
            "decode_tok_s": 0.0,
            "e2e_tok_s": 0.0,
            "total_s": round(total, 2),
        }
    # tok/s over the decode window (excludes prefill), and end-to-end
    dps = 0.0
    e2e = 0.0
    if completion_tokens is not None and decode_s > 0:
        dps = (completion_tokens - 1) / decode_s
    if completion_tokens is not None and total > 0:
        e2e = completion_tokens / total
    return {
        "label": label,
        "ttft_ms": round(ttft * 1000, 1) if ttft else None,
        "deltas": n,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "decode_tok_s": round(dps, 2),
        "e2e_tok_s": round(e2e, 2),
        "total_s": round(total, 2),
        "sample": "".join(text)[:160].replace("\n", " "),
    }


if __name__ == "__main__":
    runs = []
    runs.append(
        stream("What is the capital of France? Answer in one word.", 16, False, "known-answer")
    )
    for i in (1, 2, 3):
        runs.append(
            stream(
                "Write a Python function that merges two sorted lists. Code only.",
                256,
                False,
                f"code-256 run{i}",
            )
        )
    runs.append(
        stream(
            "Summarize this in one sentence: "
            + ("the quick brown fox jumps over the lazy dog. " * 900),
            64,
            False,
            "deep-prompt",
        )
    )
    for r in runs:
        print(json.dumps(r))
