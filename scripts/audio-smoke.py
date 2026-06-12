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
import urllib.request
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
        help="Override STT URL for direct Parakeet testing (optional)",
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

    # Test 4 (optional): Parakeet STT directly, when --stt-url is given. This is
    # the issue-#39 repro against the backend itself, bypassing the facade.
    if args.stt_url:
        print(f"\nTesting Parakeet STT directly at {args.stt_url}")
        results.append(("stt-direct", check_transcription(args.stt_url)))

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
