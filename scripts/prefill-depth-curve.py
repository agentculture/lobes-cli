#!/usr/bin/env python3
"""Measure the PREFILL-vs-DEPTH curve of an OpenAI-compatible lane (stdlib only).

WHY THIS EXISTS
---------------
A single prefill number is misleading. ``llama-bench -p 512`` and vLLM's own
short-prompt timings measure prefill at *shallow* depth, where per-request fixed
costs dominate and attention cost is negligible. Real callers paste long
documents, and prefill throughput DECAYS with depth.

Measured on a Jetson AGX Orin (sm_87) serving Qwen3.8-27B UD-Q4_K_M via
llama.cpp at MAXN, 2026-08-23:

    depth      instantaneous prefill
      2 048    240 tok/s
     84 000    148 tok/s
     98 000    139 tok/s
    114 688    129 tok/s

Using the shallow number (254 tok/s from ``pp512``) to predict TTFT at 115K
tokens understates it by ~2x. The curve — not the point — is the honest artifact.

The decay flattens rather than collapsing (98K -> 115K lost only 7%, versus 42%
over the first 98K), so deep context stays usable at a predictable premium. That
shape is worth knowing per model and per box, because it is what tells you where
a lane stops being interactive.

WHAT IT MEASURES
----------------
For each requested depth it reports, from the SERVER's own accounting where
available and from wall-clock otherwise:

  prompt_tokens   the depth the server actually saw (never the requested figure)
  prefill         tokens/s for the prompt
  ttft            time to first CONTENT token, as a streaming client sees it
  decode          tokens/s after the first token

USAGE
-----
    python3 scripts/prefill-depth-curve.py --url http://127.0.0.1:8090 \\
        --depths 512,4096,16384,65536 --model cortex

    # through a lobes gateway (add the key if GATEWAY_API_KEY is set)
    python3 scripts/prefill-depth-curve.py --url http://127.0.0.1:8000 \\
        --model cortex --api-key "$LOBES_KEY"

Deep depths are SLOW by construction — a 100K-token prefill is minutes. Start
shallow and add depths as you learn the curve. ``--max-seconds`` bounds each
request so one hung depth cannot stall the run.

Emit ``--json`` to append the run to an evidence transcript.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# A neutral filler sentence. Deliberately mundane: the point is to occupy
# context, not to bias the model toward any topic.
_FILLER = (
    "The mesh serves models across boxes. Each lobe has a role and a budget. "
    "Telemetry is scraped periodically. Contexts are measured, not assumed. "
).split()


def _haystack(approx_tokens: int) -> str:
    """Build filler of roughly ``approx_tokens`` tokens.

    Words are a rough proxy for tokens (~1.3 tokens/word on this tokenizer), so
    the SERVER's reported ``prompt_tokens`` is what the report cites — never this
    estimate.
    """
    words = [_FILLER[i % len(_FILLER)] for i in range(int(approx_tokens / 1.3))]
    return " ".join(words)


def measure(
    url: str, model: str, depth: int, api_key: str | None, max_seconds: float, gen_tokens: int
) -> dict:
    """One depth: stream a short generation and time it."""
    prompt = (_haystack(depth) + "\n\n" if depth else "") + "Reply with the single word OK."
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen_tokens,
        "temperature": 0,
        "stream": True,
        # thinking off: we are timing the transport, not the reasoning trace
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers=headers,
    )

    t0 = time.perf_counter()
    ttft = None
    n_content = 0
    last = t0
    usage = {}
    try:
        with urllib.request.urlopen(req, timeout=max_seconds) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except ValueError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                if delta.get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    n_content += 1
                    last = time.perf_counter()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"depth_requested": depth, "error": str(exc)}

    if ttft is None:
        return {"depth_requested": depth, "error": "no content tokens streamed"}

    gen_window = max(last - t0 - ttft, 1e-9)
    prompt_tokens = usage.get("prompt_tokens")
    return {
        "depth_requested": depth,
        # the server's own count, NOT our word-based estimate
        "prompt_tokens": prompt_tokens,
        "ttft_ms": round(ttft * 1000, 1),
        # prefill is only meaningful when the server told us the real depth
        "prefill_tok_s": (round(prompt_tokens / ttft, 1) if prompt_tokens and ttft else None),
        "decode_tok_s": round(n_content / gen_window, 2),
        "generated": n_content,
        "wall_s": round(last - t0, 2),
    }


def measure_generation(
    url: str,
    model: str,
    gen_tokens: int,
    api_key: str | None,
    max_seconds: float,
    sample_every: int = 256,
) -> dict:
    """One long generation, sampling the decode rate as the output grows.

    This is a DIFFERENT axis from prompt depth. The KV cache also grows while
    the model generates, so a rate measured over a 128-token burst does not
    predict a 4000-token answer. This is what tells you the SANE MAXIMUM
    generation length for a lane — the point past which decode has decayed
    enough that a caller would rather chunk the work.
    """
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a long, detailed technical essay about distributed "
                    "systems. Keep writing until you are told to stop."
                ),
            }
        ],
        "max_tokens": gen_tokens,
        "temperature": 0.7,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers=headers,
    )

    t0 = time.perf_counter()
    ttft = None
    n = 0
    samples = []
    prev_n, prev_t = 0, None
    try:
        with urllib.request.urlopen(req, timeout=max_seconds) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except ValueError:
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                if not delta.get("content"):
                    continue
                now = time.perf_counter()
                if ttft is None:
                    ttft, prev_t = now - t0, now
                n += 1
                if n - prev_n >= sample_every:
                    # instantaneous rate over this window, not a running average
                    samples.append(
                        {
                            "at_token": n,
                            "window_tok_s": round((n - prev_n) / (now - prev_t), 2),
                        }
                    )
                    prev_n, prev_t = n, now
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"generated": n, "error": str(exc), "samples": samples}

    total = time.perf_counter() - t0
    first = samples[0]["window_tok_s"] if samples else None
    last = samples[-1]["window_tok_s"] if samples else None
    return {
        "requested_tokens": gen_tokens,
        "generated": n,
        "ttft_ms": round(ttft * 1000, 1) if ttft else None,
        "mean_decode_tok_s": round(n / max(total - (ttft or 0), 1e-9), 2),
        "first_window_tok_s": first,
        "last_window_tok_s": last,
        "decay_pct": (round(100 * (1 - last / first), 1) if first and last else None),
        "samples": samples,
        "wall_s": round(total, 2),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Measure prefill/TTFT/decode against prompt depth.")
    ap.add_argument(
        "--url", required=True, help="lane or gateway origin, e.g. http://127.0.0.1:8090"
    )
    ap.add_argument(
        "--model", default="cortex", help="model or role alias to address (default: cortex)"
    )
    ap.add_argument(
        "--depths",
        default="0,512,2048,8192,32768",
        help="comma-separated approximate prompt depths",
    )
    ap.add_argument("--api-key", default=None, help="bearer token, if the gateway gate is armed")
    ap.add_argument(
        "--max-seconds", type=float, default=3600.0, help="per-request ceiling (default 3600)"
    )
    ap.add_argument(
        "--gen-tokens", type=int, default=32, help="tokens to generate per depth (default 32)"
    )
    ap.add_argument(
        "--gen-sweep",
        type=int,
        default=0,
        metavar="N",
        help="also measure decode decay over an N-token generation "
        "(the SANE-MAX axis); 0 disables",
    )
    ap.add_argument(
        "--gen-sample-every",
        type=int,
        default=256,
        help="sample the generation rate every N tokens",
    )
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args(argv)

    depths = [int(d) for d in args.depths.split(",") if d.strip()]
    rows = []
    for depth in depths:
        row = measure(args.url, args.model, depth, args.api_key, args.max_seconds, args.gen_tokens)
        rows.append(row)
        if not args.json:
            if "error" in row:
                print(f"  depth {depth:>7}: FAILED — {row['error']}")
            else:
                print(
                    f"  depth {depth:>7}: "
                    f"prompt_tokens {str(row['prompt_tokens']):>7} | "
                    f"ttft {row['ttft_ms']:>9.1f} ms | "
                    f"prefill {str(row['prefill_tok_s']):>7} tok/s | "
                    f"decode {row['decode_tok_s']:>6.2f} tok/s"
                )
            sys.stdout.flush()

    gen = None
    if args.gen_sweep:
        if not args.json:
            print(
                f"\n  generation sweep: {args.gen_sweep} tokens "
                f"(sampling every {args.gen_sample_every})"
            )
            sys.stdout.flush()
        gen = measure_generation(
            args.url,
            args.model,
            args.gen_sweep,
            args.api_key,
            args.max_seconds,
            args.gen_sample_every,
        )
        if not args.json:
            if "error" in gen:
                print(f"    FAILED after {gen['generated']} tokens - {gen['error']}")
            else:
                print(
                    f"    generated {gen['generated']} tok in {gen['wall_s']} s, "
                    f"mean {gen['mean_decode_tok_s']} tok/s"
                )
                for smp in gen["samples"]:
                    print(
                        f"      at {smp['at_token']:>6} tok: " f"{smp['window_tok_s']:>6.2f} tok/s"
                    )
                if gen.get("decay_pct") is not None:
                    print(
                        f"    decode decay over the generation: {gen['decay_pct']}% "
                        f"({gen['first_window_tok_s']} -> {gen['last_window_tok_s']} tok/s)"
                    )

    if args.json:
        out = {"url": args.url, "model": args.model, "rows": rows}
        if gen is not None:
            out["generation_sweep"] = gen
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
