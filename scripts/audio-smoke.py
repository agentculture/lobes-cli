#!/usr/bin/env python3
"""Smoke test for the live audio realtime surface.

Tests the OpenAI /v1/audio/* endpoints (transcriptions + speech) against a
running model-gear fleet with the audio overlay. This is NOT an offline CI test
— it requires a live GPU box with `model fleet up` already active.

Reproduces the issue #39 repro to confirm the 500→200 fix: generates a 2s 440 Hz
tone (16 kHz mono PCM16 WAV), posts it to /v1/audio/transcriptions, and asserts
HTTP 200 + a JSON response with a `text` key. Also exercises /v1/audio/speech
(Magpie TTS) and, when --stt-url is given, the Parakeet backend directly.

Exit code 0 if all checks pass; non-zero on any failure.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
import urllib.request
import wave
from urllib.error import URLError


def build_wav_tone(
    duration_sec: float = 2.0, freq_hz: float = 440.0, sample_rate: int = 16000
) -> bytes:
    """Generate a sine-wave WAV file (16-bit PCM mono).

    Args:
        duration_sec: length in seconds.
        freq_hz: frequency in Hz (default 440 A4).
        sample_rate: samples per second (default 16000 for speech).

    Returns:
        Complete WAV file (RIFF header + PCM samples).
    """
    num_samples = int(duration_sec * sample_rate)
    # Generate sine samples (16-bit signed).
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        value = int(32767 * 0.5 * math.sin(2 * math.pi * freq_hz * t))
        # Clamp to 16-bit range.
        value = max(-32768, min(32767, value))
        samples.append(value)

    # Pack as little-endian 16-bit signed integers.
    data = b"".join(v.to_bytes(2, byteorder="little", signed=True) for v in samples)

    # WAV header (RIFF container, PCM format).
    byte_rate = sample_rate * 2  # 2 bytes per sample (16-bit).
    block_align = 2
    subchunk2_size = len(data)
    chunk_size = 36 + subchunk2_size

    wav = io.BytesIO()
    wav.write(b"RIFF")
    wav.write(chunk_size.to_bytes(4, byteorder="little"))
    wav.write(b"WAVE")

    wav.write(b"fmt ")
    wav.write((16).to_bytes(4, byteorder="little"))  # subchunk1_size
    wav.write((1).to_bytes(2, byteorder="little"))  # audio_format (PCM)
    wav.write((1).to_bytes(2, byteorder="little"))  # num_channels (mono)
    wav.write(sample_rate.to_bytes(4, byteorder="little"))
    wav.write(byte_rate.to_bytes(4, byteorder="little"))
    wav.write(block_align.to_bytes(2, byteorder="little"))
    wav.write((16).to_bytes(2, byteorder="little"))  # bits_per_sample

    wav.write(b"data")
    wav.write(subchunk2_size.to_bytes(4, byteorder="little"))
    wav.write(data)

    return wav.getvalue()


def check_openapi(base_url: str) -> bool:
    """Check that GET /openapi.json lists both /v1/audio endpoints.

    Args:
        base_url: base URL (e.g., http://localhost:8080).

    Returns:
        True if both endpoints are listed; False otherwise.
    """
    url = f"{base_url.rstrip('/')}/openapi.json"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            if resp.status != 200:
                print(f"FAIL: openapi.json returned {resp.status}")
                return False
            body = json.loads(resp.read().decode("utf-8"))
    except (URLError, json.JSONDecodeError) as exc:
        print(f"FAIL: openapi.json request failed: {exc}")
        return False

    paths = body.get("paths", {})
    has_transcriptions = "/v1/audio/transcriptions" in paths
    has_speech = "/v1/audio/speech" in paths

    if has_transcriptions and has_speech:
        print("PASS: openapi.json lists /v1/audio/transcriptions and /v1/audio/speech")
        return True
    else:
        print(
            f"FAIL: openapi.json missing endpoints "
            f"(transcriptions={has_transcriptions}, speech={has_speech})"
        )
        return False


def check_transcription(base_url: str) -> bool:
    """Test POST /v1/audio/transcriptions with a 2s 440 Hz tone.

    Args:
        base_url: base URL (e.g., http://localhost:8080).

    Returns:
        True if the endpoint returns 200 + valid JSON with 'text' key; False otherwise.
    """
    url = f"{base_url.rstrip('/')}/v1/audio/transcriptions"
    wav_data = build_wav_tone(duration_sec=2.0, freq_hz=440.0, sample_rate=16000)

    boundary = "----WebKitFormBoundary"
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="file"; filename="test.wav"\r\n')
    body.write(b"Content-Type: audio/wav\r\n\r\n")
    body.write(wav_data)
    body.write(f"\r\n--{boundary}--\r\n".encode())

    req = urllib.request.Request(url, data=body.getvalue())
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                print(f"FAIL: transcriptions returned {resp.status}")
                return False
            response_body = json.loads(resp.read().decode("utf-8"))
    except (URLError, json.JSONDecodeError) as exc:
        print(f"FAIL: transcriptions request failed: {exc}")
        return False

    if "text" in response_body:
        text = response_body.get("text", "")
        print(f"PASS: transcriptions returned 200 with text='{text}'")
        return True
    else:
        print(f"FAIL: transcriptions response missing 'text' key: {response_body}")
        return False


def check_speech(base_url: str) -> bool:
    """Test POST /v1/audio/speech (OpenAI TTS → Magpie) returns audio.

    Args:
        base_url: base URL (e.g., http://localhost:8080).

    Returns:
        True if the endpoint returns 200 with a non-empty audio body; else False.
    """
    url = f"{base_url.rstrip('/')}/v1/audio/speech"
    payload = json.dumps({"input": "hey reachy", "voice": "Mia.Calm"}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                print(f"FAIL: speech returned {resp.status}")
                return False
            content_type = resp.headers.get("Content-Type", "")
            body_len = len(resp.read())
    except URLError as exc:
        print(f"FAIL: speech request failed: {exc}")
        return False

    if body_len > 0 and "audio" in content_type:
        print(f"PASS: speech returned 200 ({body_len} bytes, {content_type})")
        return True
    else:
        print(
            f"FAIL: speech returned 200 but body/type unexpected "
            f"(bytes={body_len}, content_type={content_type!r})"
        )
        return False


def check_round_trip(chatterbox_url: str, stt_url: str) -> bool:
    """Test Chatterbox TTS → WAV → Parakeet STT round-trip with 3 known phrases.

    Synthesizes each phrase via ``POST {chatterbox_url}/v1/audio/synthesize``
    (JSON ``{"text": ...}`` body), wraps the returned raw PCM16 bytes in a
    24 kHz mono WAV using stdlib :mod:`wave`, then POSTs that WAV to
    ``{stt_url}/v1/audio/transcriptions`` and checks that key words from the
    original phrase appear in the transcription (case-insensitive).

    Keyword lists account for documented Parakeet normalizations:
    - spoken numerals → digits (e.g. "eight thousand and one" → "8001")
    - proper nouns rendered phonetically (e.g. "Reachy" → "Ricci")

    Args:
        chatterbox_url: Base URL of the Chatterbox sidecar (e.g. http://localhost:9100).
        stt_url: Base URL of the Parakeet STT service (e.g. http://localhost:9002).

    Returns:
        True if all phrases round-trip successfully; False if any fail.
    """
    # Each entry: (phrase text, required keywords in transcription).
    # Keywords are chosen to survive Parakeet's numeric/phonetic normalizations.
    phrases = [
        (
            "The quick brown fox jumps over the lazy dog.",
            ["quick", "brown", "fox", "lazy", "dog"],
        ),
        (
            "Can you reach the gateway on port eight thousand and one?",
            # Parakeet normalises "eight thousand and one" → "8001"
            ["gateway", "port", "8001"],
        ),
        (
            "Reachy is online and ready.",
            # Parakeet renders "Reachy" phonetically and may split "online" → "on line";
            # check only "ready" which is always faithfully transcribed
            ["ready"],
        ),
    ]

    synth_url = f"{chatterbox_url.rstrip('/')}/v1/audio/synthesize"
    trans_url = f"{stt_url.rstrip('/')}/v1/audio/transcriptions"
    boundary = "----ModelGearRoundTrip"

    all_ok = True
    for text, keywords in phrases:
        # Step 1: synthesize → raw PCM16
        payload = json.dumps({"text": text}).encode("utf-8")
        req_synth = urllib.request.Request(synth_url, data=payload, method="POST")
        req_synth.add_header("Content-Type", "application/json")
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req_synth, timeout=180) as resp:
                if resp.status != 200:
                    print(f"FAIL round-trip: synthesize returned {resp.status} for {text!r}")
                    all_ok = False
                    continue
                pcm = resp.read()
        except URLError as exc:
            print(f"FAIL round-trip: synthesize error for {text!r}: {exc}")
            all_ok = False
            continue
        latency = time.monotonic() - t0

        # Step 2: wrap PCM16 in a 24 kHz mono WAV container
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)   # 16-bit
            wf.setframerate(24000)
            wf.writeframes(pcm)
        wav_data = wav_buf.getvalue()

        # Step 3: transcribe WAV → text
        body = io.BytesIO()
        body.write(f"--{boundary}\r\n".encode())
        body.write(
            b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
        )
        body.write(b"Content-Type: audio/wav\r\n\r\n")
        body.write(wav_data)
        body.write(f"\r\n--{boundary}--\r\n".encode())

        req_trans = urllib.request.Request(trans_url, data=body.getvalue(), method="POST")
        req_trans.add_header(
            "Content-Type", f"multipart/form-data; boundary={boundary}"
        )
        try:
            with urllib.request.urlopen(req_trans, timeout=60) as resp:
                if resp.status != 200:
                    print(
                        f"FAIL round-trip: transcriptions returned {resp.status} "
                        f"for {text!r}"
                    )
                    all_ok = False
                    continue
                transcription = json.loads(resp.read().decode("utf-8")).get("text", "")
        except (URLError, json.JSONDecodeError) as exc:
            print(f"FAIL round-trip: transcriptions error for {text!r}: {exc}")
            all_ok = False
            continue

        trans_lower = transcription.lower()
        missed = [kw for kw in keywords if kw.lower() not in trans_lower]
        if missed:
            print(
                f"FAIL round-trip: phrase={text!r} | "
                f"transcription={transcription!r} | missing={missed}"
            )
            all_ok = False
        else:
            print(
                f"PASS round-trip: phrase={text!r} | "
                f"latency={latency:.2f}s | pcm={len(pcm)}B | "
                f"transcription={transcription!r}"
            )

    return all_ok


def main() -> int:
    """Run all smoke tests.

    Returns:
        0 if all tests pass; 1 if any fail.
    """
    parser = argparse.ArgumentParser(
        description="Smoke test the model-gear audio realtime surface."
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080",
        help="Base URL of the realtime service (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--stt-url",
        default="http://localhost:9002",
        help="Parakeet STT URL for direct testing (default: http://localhost:9002)",
    )
    parser.add_argument(
        "--chatterbox-url",
        default="http://localhost:9100",
        help="Chatterbox TTS sidecar URL for round-trip check (default: http://localhost:9100)",
    )
    args = parser.parse_args()

    print(f"Testing audio surface at {args.base_url}")
    print()

    results = []

    # Test 1: OpenAPI schema
    results.append(("openapi.json", check_openapi(args.base_url)))

    # Test 2: Transcription endpoint (facade → Parakeet)
    results.append(("transcriptions", check_transcription(args.base_url)))

    # Test 3: Speech endpoint (facade → Magpie)
    results.append(("speech", check_speech(args.base_url)))

    # Test 4 (optional): Parakeet STT directly, bypassing the facade.
    if args.stt_url and args.stt_url != "http://localhost:9002":
        print(f"\nTesting Parakeet STT directly at {args.stt_url}")
        results.append(("stt-direct", check_transcription(args.stt_url)))

    # Test 5 (optional): Chatterbox→Parakeet round-trip. Only runs when the
    # Chatterbox sidecar is reachable (skip gracefully otherwise, like
    # stt-direct above).
    try:
        health_url = f"{args.chatterbox_url.rstrip('/')}/v1/health/ready"
        with urllib.request.urlopen(health_url, timeout=3) as _r:
            chatterbox_up = _r.status == 200
    except URLError:
        chatterbox_up = False

    if chatterbox_up:
        print(f"\nTesting Chatterbox→Parakeet round-trip")
        print(f"  Chatterbox: {args.chatterbox_url}")
        print(f"  STT:        {args.stt_url}")
        results.append(
            ("round-trip", check_round_trip(args.chatterbox_url, args.stt_url))
        )
    else:
        print(
            f"\nSKIP: Chatterbox sidecar not reachable at {args.chatterbox_url} "
            f"(start with: CHATTERBOX_PORT=9100 python -m model_gear.realtime.chatterbox_server)"
        )

    print()
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Results: {passed}/{total} checks passed")

    if passed == total:
        print("SUCCESS: all audio surface checks passed")
        return 0
    else:
        print("FAILURE: some checks failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
