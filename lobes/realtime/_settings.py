"""Realtime settings — env → a frozen :class:`Settings` (stdlib only).

Replaces the sibling project's pydantic-settings ``config.py``. lobes keeps
config parsing in the standard library so this module is importable and testable
without the ``[realtime]`` extra (mirrors :mod:`lobes.gateway._config`).

The env keys are the field names upper-cased — set by the ``realtime`` fleet
service's ``environment:`` block. A module-level ``settings`` singleton is built
from ``os.environ`` at import (the container's env); tests call
:func:`build_settings` with an explicit mapping or pass values explicitly.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Where the realtime service finds its backends + its audio defaults."""

    # Backend service URLs (reachable on the fleet compose network).
    tts_url: str  # Chatterbox TTS sidecar, e.g. http://chatterbox:9000
    stt_url: str  # Parakeet STT, e.g. http://stt:9002
    openai_base_url: str  # the fleet LLM front, e.g. http://gateway:8000
    openai_api_key: str
    openai_model: str  # may be "" → the gateway default-routes

    # TTS defaults.
    default_voice: str  # "" → Chatterbox default voice; or a .wav path for zero-shot cloning
    tts_speed: int
    tts_concurrency: int  # max parallel TTS requests (1 = serial)

    # VAD / turn detection (used by the realtime WS pipeline).
    vad_threshold: float
    vad_silence_ms: int
    vad_prefix_padding_ms: int
    vad_max_turn_ms: int
    default_turn_detection: str
    default_aec_mode: str
    barge_in_window_ms: int
    barge_in_model: str | None

    # Where the FastAPI app listens (inside the container).
    host: str
    port: int


def _as_int(env: Mapping[str, str], key: str, default: int) -> int:
    try:
        return int(env.get(key) or default)
    except (TypeError, ValueError):
        return int(default)


def _as_float(env: Mapping[str, str], key: str, default: float) -> float:
    try:
        return float(env.get(key) or default)
    except (TypeError, ValueError):
        return float(default)


def build_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Construct :class:`Settings` from environment variables (pure)."""
    env = os.environ if env is None else env
    return Settings(
        tts_url=(env.get("TTS_URL") or "http://chatterbox:9000").rstrip("/"),
        stt_url=(env.get("STT_URL") or "http://stt:9002").rstrip("/"),
        openai_base_url=(env.get("OPENAI_BASE_URL") or "http://gateway:8000").rstrip("/"),
        openai_api_key=env.get("OPENAI_API_KEY") or "EMPTY",
        openai_model=env.get("OPENAI_MODEL") or "",
        default_voice=env.get("DEFAULT_VOICE") or "",
        # Clamp to >=1: tts_concurrency seeds an asyncio.Semaphore, and Semaphore(0)
        # (or negative) blocks every TTS request forever; tts_speed is a percentage,
        # where 0/negative would emit nonsensical SSML rate="0%" → backend 502s.
        tts_speed=max(1, _as_int(env, "TTS_SPEED", 125)),
        tts_concurrency=max(1, _as_int(env, "TTS_CONCURRENCY", 1)),
        vad_threshold=_as_float(env, "VAD_THRESHOLD", 0.5),
        vad_silence_ms=_as_int(env, "VAD_SILENCE_MS", 600),
        vad_prefix_padding_ms=_as_int(env, "VAD_PREFIX_PADDING_MS", 300),
        # VAD_MAX_TURN_MS: hard cap on one uninterrupted turn before the
        # segmenter force-commits it (lobes.realtime._segmenter's
        # DEFAULT_MAX_TURN_MS=30_000 — same default, now env-tunable). Wired
        # through docker-compose.audio.yml / env.audio.example in #149 t5;
        # this is the settings-side read t5 left for t6 to add.
        vad_max_turn_ms=_as_int(env, "VAD_MAX_TURN_MS", 30_000),
        default_turn_detection=env.get("DEFAULT_TURN_DETECTION") or "server_vad",
        default_aec_mode=env.get("DEFAULT_AEC_MODE") or "none",
        barge_in_window_ms=_as_int(env, "BARGE_IN_WINDOW_MS", 750),
        barge_in_model=env.get("BARGE_IN_MODEL") or None,
        host=env.get("REALTIME_HOST") or "0.0.0.0",  # nosec B104 — bind all inside the container
        port=_as_int(env, "REALTIME_PORT", 8080),
    )


# The container's live settings (env set by the realtime compose service).
settings = build_settings()
