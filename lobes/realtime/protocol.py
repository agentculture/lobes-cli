"""Event ID generation and audio constants for the realtime pipeline.

Stdlib-only (time, uuid, enum) so it imports without the ``[realtime]`` extra.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum

# ---------------------------------------------------------------------------
# Audio constants
# ---------------------------------------------------------------------------
CLIENT_SAMPLE_RATE = 24000  # OpenAI Realtime API uses 24kHz PCM16
TTS_SAMPLE_RATE = 24000  # Chatterbox outputs 24000Hz — matches CLIENT_SAMPLE_RATE, no resample
STT_SAMPLE_RATE = 16000  # Parakeet expects 16kHz
VAD_SAMPLE_RATE = 16000  # Silero VAD expects 16kHz
BYTES_PER_SAMPLE = 2  # 16-bit PCM

# Silero VAD requires 512 samples at 16kHz (32ms chunks)
VAD_CHUNK_SAMPLES = 512
VAD_CHUNK_MS = 32


class AudioFormat(str, Enum):
    PCM16 = "pcm16"


class TurnDetectionType(str, Enum):
    SERVER_VAD = "server_vad"


class AECMode(str, Enum):
    NONE = "none"
    AEC = "aec"


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------
def gen_event_id() -> str:
    return f"event_{uuid.uuid4().hex[:24]}"


def gen_item_id() -> str:
    return f"item_{uuid.uuid4().hex[:24]}"


def gen_response_id() -> str:
    return f"resp_{uuid.uuid4().hex[:24]}"


def gen_session_id() -> str:
    return f"sess_{uuid.uuid4().hex[:24]}"


def gen_content_part_id() -> str:
    return f"part_{uuid.uuid4().hex[:24]}"


def timestamp_ms() -> int:
    return int(time.monotonic() * 1000)


# ---------------------------------------------------------------------------
# Voice resolution: Chatterbox sidecar
# ---------------------------------------------------------------------------


def resolve_voice(voice: str) -> str:
    """Resolve a voice selector for the Chatterbox TTS sidecar.

    Chatterbox ships one default voice; zero-shot cloning is activated by
    supplying a reference ``.wav`` file path.

    - ``voice`` ending in ``.wav`` → returned as-is (passed as
      ``audio_prompt_path`` to ``ChatterboxTTS.generate``).
    - Any other value (OpenAI name, empty string, …) → ``""`` (default voice).
    """
    if voice.lower().endswith(".wav"):
        return voice
    return ""
