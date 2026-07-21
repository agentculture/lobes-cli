#!/usr/bin/env python3
"""Talk to this machine, voice to voice, over the lobes fleet.

`/v1/realtime` shipped as EARS only — audio in, speech boundaries and
transcripts out (spec non-goal c15: no response.create, no LLM turn, no TTS
over the session). A conversation therefore lives in the client, stitching
three endpoints that are all served by this box:

    ears   ws  /v1/realtime                 Silero VAD + Parakeet
    brain  POST /v1/chat/completions         Gemma 4 12B (model=multimodal)
    mouth  POST /v1/audio/speech             Chatterbox TTS

No barge-in: the loop stops listening while it speaks, because without echo
cancellation the mic would hear the speakers and transcribe the machine
talking to itself.

Scratch tool. Reuses the SHIPPED smoke script's WebSocket client so the
protocol path exercised here is the committed one.

  python3 voice-loop.py --device hw:2,0 --api-key "$KEY"
"""

from __future__ import annotations

import argparse
import array
import importlib.util
import json
import os
import queue
import socket
import subprocess
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

SMOKE = Path(__file__).resolve().parent / "realtime-smoke.py"
_spec = importlib.util.spec_from_file_location("realtime_smoke", SMOKE)
rs = importlib.util.module_from_spec(_spec)
sys.modules["realtime_smoke"] = rs
_spec.loader.exec_module(rs)

# Generous enough for a long spoken reply, short enough that a wedged audio
# backend cannot strand the conversation.
PLAYBACK_TIMEOUT_S = 60

RATE = 16000
CHUNK_BYTES = RATE * 2 * 32 // 1000  # 32 ms of PCM16 mono = 1024 bytes

SYSTEM_PROMPT = (
    "You are the voice of this machine — a DGX Spark running the lobes fleet. "
    "You are being spoken to out loud and your reply is read back aloud by a "
    "text-to-speech voice, so answer in one or two short spoken sentences. "
    "No markdown, no lists, no code blocks, no emoji — just what you would say."
)


def _post_json(url: str, payload: dict, api_key: str | None, timeout: float = 120.0) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310 - operator-supplied URL
        return json.loads(resp.read())


def think(base_url: str, api_key: str | None, history: list[dict], said: str, model: str) -> str:
    history.append({"role": "user", "content": said})
    out = _post_json(
        base_url.rstrip("/") + "/v1/chat/completions",
        {
            # The Gemma 4 12B lane, NOT cortex: measured ~1s to first reply on
            # this box vs the 27B's reasoning latency. A spoken turn is dead
            # air until the answer starts, so speed beats depth here.
            "model": model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + history,
            "max_tokens": 160,
            "temperature": 0.7,
            # No thinking: a reasoning trace is latency the speaker cannot use.
            "chat_template_kwargs": {"enable_thinking": False},
        },
        api_key,
    )
    reply = (out["choices"][0]["message"].get("content") or "").strip()
    history.append({"role": "assistant", "content": reply})
    return reply


def speak(base_url: str, api_key: str | None, text: str, sink: str | None = None) -> None:
    """Synthesize with Chatterbox and play it out.

    Playback targets the HDMI sink explicitly. The Reachy Mini speaker is
    NOT usable: reachy-mini-dae holds /dev/snd/pcmC1D0p exclusively, and
    PipeWire's node for that sink times out while the daemon owns it (both
    paplay and pw-play hang). HDMI is a separate card and plays fine —
    confirmed audible by the operator.
    """
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/audio/speech",
        data=json.dumps({"model": "tts", "input": text, "response_format": "wav"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=180) as resp:  # nosec B310
        wav = resp.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
        fh.write(wav)
        tmp = Path(fh.name)
    players = [["paplay", f"--device={sink}", str(tmp)]] if sink else []
    players += [["paplay", str(tmp)], ["pw-play", str(tmp)], ["aplay", "-q", str(tmp)]]
    for player in players:
        # Timeout is mandatory, not defensive: paplay was OBSERVED hanging on a
        # sink whose ALSA device another process held exclusively. With the mic
        # muted for the duration, a hang here deafens the session forever.
        try:
            r = subprocess.run(player, capture_output=True, timeout=PLAYBACK_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            print(f"  [playback timed out via {player[0]} — trying the next backend]", flush=True)
            continue
        if r.returncode == 0:
            tmp.unlink(missing_ok=True)
            return
    print("  [playback FAILED on every backend]", flush=True)
    tmp.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:8001")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--device", default="default", help="ALSA capture device, e.g. hw:1,0")
    ap.add_argument(
        "--channels",
        type=int,
        default=1,
        help="mic capture channels; 2 is downmixed to mono (the Reachy mic is stereo-only)",
    )
    ap.add_argument(
        "--model",
        default="multimodal",
        help="generate lane for the reply (default: the Gemma 4 12B lane — measured "
        "~1s to a short answer; a thinking model spends that on a trace nobody hears)",
    )
    ap.add_argument(
        "--sink",
        default=None,
        help="PulseAudio/PipeWire sink for playback; omit to use the default output",
    )
    ap.add_argument("--turns", type=int, default=6, help="how many exchanges before exiting")
    ap.add_argument("--idle-timeout", type=float, default=90.0)
    args = ap.parse_args()
    # Prefer the env var: argv is world-readable via /proc, so a key passed as
    # --api-key shows up in `ps` for every user on the box.
    args.api_key = args.api_key or os.environ.get("LOBES_API_KEY")

    _scheme, host, port, path = rs.build_realtime_ws_target(args.base_url, RATE)
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else None
    client, status, _hdrs = rs.WebSocketClient.connect(
        host, port, path, extra_headers=headers, connect_timeout=10.0
    )
    if status != 101:
        print(f"handshake FAILED: HTTP {status}", flush=True)
        return 1
    print(f"connected to {args.base_url}/v1/realtime — 101 Switching Protocols", flush=True)

    transcripts: queue.Queue[str] = queue.Queue()
    stop = threading.Event()

    def reader() -> None:
        try:
            _reader_loop()
        except Exception:
            import traceback

            print("  [EVENT READER DIED]", flush=True)
            traceback.print_exc()
            stop.set()

    def _reader_loop() -> None:
        while not stop.is_set():
            try:
                _fin, opcode, payload = client.read_frame(timeout=1.0)
            except socket.timeout:
                continue  # just a quiet second — keep listening
            except Exception as exc:
                # EOF / FrameReadError means the session is GONE. Retrying would
                # spin forever and leave the main loop waiting out its idle
                # timeout, which reads as "nobody spoke" rather than "the
                # connection died".
                print(f"  [session ended: {type(exc).__name__}: {exc}]", flush=True)
                stop.set()
                return
            if opcode == rs.OPCODE_PING:
                # MUST answer, or the server closes the session. uvicorn pings
                # every ~20s and drops a peer that never pongs — which is why a
                # long conversation died after tens of seconds while the
                # one-shot smoke run (well under one ping interval) never did.
                # send_frame is lock-guarded, so ponging from this thread is
                # safe alongside the mic feeder's writes.
                client.send_frame(rs.OPCODE_PONG, payload)
                continue
            if opcode == rs.OPCODE_CLOSE:
                print("  [server closed the session]", flush=True)
                stop.set()
                return
            if opcode != rs.OPCODE_TEXT or not payload:
                continue
            try:
                evt = json.loads(payload)
            except ValueError:
                continue
            kind = evt.get("type", "")
            if kind.endswith("speech_started"):
                print("  [hearing you...]", flush=True)
            elif "transcription" in kind:
                text = (evt.get("transcript") or evt.get("text") or "").strip()
                if text:
                    transcripts.put(text)
            elif kind == "error":
                print(f"  [session error] {evt}", flush=True)

    threading.Thread(target=reader, daemon=True).start()

    mic = subprocess.Popen(
        [
            "arecord",
            "-D",
            args.device,
            "-f",
            "S16_LE",
            "-r",
            str(RATE),
            "-c",
            str(args.channels),
            "-t",
            "raw",
            "-q",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    read_bytes = CHUNK_BYTES * args.channels  # stereo delivers 2x per mono chunk

    # Fail LOUDLY if the capture device could not be opened. arecord exits
    # immediately on "Device or resource busy" and its stdout just EOFs, which
    # is indistinguishable from a silent room — the loop would sit there
    # looking healthy until the idle timeout. Probe the first read instead.
    first = mic.stdout.read(read_bytes)
    if not first:
        err = (mic.stderr.read() or b"").decode(errors="replace").strip() if mic.stderr else ""
        raise SystemExit(
            f"mic device {args.device!r} produced no audio — {err or 'device unavailable'}\n"
            f"hint: something else may hold it (check: fuser -v /dev/snd/*)"
        )

    def to_mono(raw: bytes) -> bytes:
        """Left channel only. The session wants PCM16 MONO; a stereo frame fed
        straight in would read as double-rate garbage to the VAD."""
        if args.channels == 1:
            return raw
        samples = array.array("h")
        samples.frombytes(raw[: len(raw) // 4 * 4])
        return array.array("h", samples[0::2]).tobytes()

    muted = threading.Event()  # set while the machine is speaking (no barge-in, no echo)

    def feed() -> None:
        # A silently dying feeder deafens the session while the main loop waits
        # forever — surface it loudly instead.
        try:
            _feed_loop()
        except Exception:
            import traceback

            print("  [MIC FEED DIED — session is deaf]", flush=True)
            traceback.print_exc()
            stop.set()

    def _feed_loop() -> None:
        pending_first = first
        while not stop.is_set():
            if pending_first:
                raw, pending_first = pending_first, b""
            else:
                raw = mic.stdout.read(read_bytes)
            if not raw:
                print("  [mic stopped producing audio — ending session]", flush=True)
                stop.set()
                return
            chunk = to_mono(raw)
            # While speaking, send silence instead of mic audio: without AEC the
            # mic hears the speakers and the machine transcribes itself.
            client.send_binary(b"\x00" * len(chunk) if muted.is_set() else chunk)

    threading.Thread(target=feed, daemon=True).start()

    history: list[dict] = []
    print(
        "\n=== SPEAK NOW. Pause when you finish a sentence. Say 'goodbye' to end. ===\n", flush=True
    )
    try:
        for turn in range(args.turns):
            try:
                said = transcripts.get(timeout=args.idle_timeout)
            except queue.Empty:
                print("(nothing heard — ending)", flush=True)
                break
            print(f"YOU:  {said}", flush=True)
            if "goodbye" in said.lower() or "good bye" in said.lower():
                muted.set()
                speak(args.base_url, args.api_key, "Goodbye.", args.sink)
                break
            try:
                reply = think(args.base_url, args.api_key, history, said, args.model)
            except Exception as exc:  # noqa: BLE001 - a spoken loop must not die on one bad turn
                print(f"  [generate error] {exc}", flush=True)
                continue
            print(f"MACHINE: {reply}", flush=True)
            muted.set()
            try:
                speak(args.base_url, args.api_key, reply, args.sink)
            except Exception as exc:  # noqa: BLE001
                print(f"  [tts error] {exc}", flush=True)
            finally:
                while not transcripts.empty():  # drop anything captured while speaking
                    transcripts.get_nowait()
                muted.clear()
    finally:
        stop.set()
        # terminate() alone can leave arecord alive holding the capture device,
        # which makes the NEXT run fail with "Device or resource busy" — wait
        # for it, then kill it outright.
        mic.terminate()
        try:
            mic.wait(timeout=3)
        except subprocess.TimeoutExpired:
            mic.kill()
        client.close()
    print("\n=== conversation ended ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
