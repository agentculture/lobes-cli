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

import asyncio
import os
from collections.abc import Mapping
from dataclasses import dataclass

from ._session import DEFAULT_LANGUAGE as _SESSION_DEFAULT_LANGUAGE
from ._session import DEFAULT_SYSTEM_PROMPT as _SESSION_DEFAULT_SYSTEM_PROMPT

# The Hebrew voice-lane default system prompt (issue #151 t8 / hebrew-realtime
# c30). Selected instead of :data:`_SESSION_DEFAULT_SYSTEM_PROMPT` ONLY when
# an operator sets REALTIME_LANGUAGE=he and leaves DEFAULT_SYSTEM_PROMPT
# unset — an explicit DEFAULT_SYSTEM_PROMPT always wins outright, exactly
# like the English default it sits beside. Written IN Hebrew (the model is
# instructed to answer in Hebrew, not merely told about Hebrew in English) —
# Chatterbox reads the reply aloud verbatim, so the same "short spoken
# sentences, no markdown" discipline as the English default applies, plus one
# Hebrew-specific instruction the English prompt has no need for: a tool
# result routinely contains file paths, identifiers, hashes, dates or long
# numbers, and reading those aloud character-by-character produces
# unintelligible/unnatural speech (worse in Hebrew, which has no standard
# spoken convention for reading raw hex or dotted paths) — so the model must
# DESCRIBE such values (how many there are, what kind, the meaningful part of
# a name) instead of reciting them verbatim.
DEFAULT_SYSTEM_PROMPT_HE = (
    "את/ה הקול של המכונה הזו. פונים אליך בעל פה והתשובה שלך מוקראת בקול "
    "רם על ידי מנוע טקסט-לדיבור, אז ענה/עני במשפט קצר אחד או שניים, "
    "בעברית מדוברת בלבד. בלי מרקדאון, בלי רשימות, בלי קוד, בלי אמוג'ים — "
    "רק מה שהיית אומר/ת בקול. כשתוצאה של כלי מכילה נתיבים, מזהים, "
    "hash-ים, תאריכים או מספרים ארוכים — תאר/י אותם במקום להקריא אותם "
    "תו-אחר-תו: כמה יש, מאיזה סוג, והחלק המשמעותי בשם."
)


@dataclass(frozen=True)
class Settings:
    """Where the realtime service finds its backends + its audio defaults."""

    # Backend service URLs (reachable on the fleet compose network).
    tts_url: str  # Chatterbox TTS sidecar, e.g. http://chatterbox:9000
    stt_url: str  # Parakeet STT, e.g. http://stt:9002
    openai_base_url: str  # the fleet LLM front, e.g. http://gateway:8000
    openai_api_key: str
    openai_model: str  # may be "" → the gateway default-routes
    # The session-default STT/generate language (issue #151 t8 / hebrew-realtime
    # c30, c3): mirrors _session.DEFAULT_LANGUAGE's own shape check (a
    # deployment-wide default, overridable per-session exactly like every
    # other _session default). Never validated here — an unsupported code is
    # the STT/generate backend's problem, not this module's; this field only
    # carries the operator's env-set deployment default through.
    language: str
    # Deadline (ms) a committed turn's tool-call round trip is allowed before
    # it force-fails (issue #151 t8, spec c3: "tool-wait deadline knob"). Read
    # but not yet consumed by any caller in this task — a later task wires it
    # into the floor's per-stage deadline machinery (mirrors how
    # VAD_MAX_TURN_MS landed as a settings-side read in #149 t6 before a
    # later task consumed it). Clamped like every other millisecond knob in
    # this module: a non-positive value would force-fail a tool call
    # immediately rather than merely doing nothing.
    tool_wait_timeout_ms: int
    # Operator-set DEFAULT system prompt for a session's generate calls (issue
    # #151 t8) — the OPERATOR half of "an operator-set default system prompt
    # via env and a per-session override in the connect config" (spec c34).
    # The per-session override half already exists as
    # SessionConfig.system_prompt / parse_session_config's own
    # ``system_prompt`` payload key, which always wins over this value when a
    # client sets one (see _session.parse_session_config's docstring). Never
    # blank: an unset env mirrors _session.DEFAULT_SYSTEM_PROMPT, so a
    # deployment that never sets DEFAULT_SYSTEM_PROMPT behaves exactly like
    # the pre-#151 in-code fallback. A genuinely blank system prompt is
    # actively harmful here, not merely useless — Chatterbox reads the reply
    # aloud verbatim, so a reply that reverts to the model's normal WRITTEN
    # register (markdown, bullet lists, code fences) comes back as spoken
    # nonsense ("asterisk asterisk", literal list markers) instead of a
    # sentence a listener can follow.
    default_system_prompt: str

    # TTS defaults.
    default_voice: str  # "" → Chatterbox default voice; or a .wav path for zero-shot cloning
    tts_speed: int
    tts_concurrency: int  # max parallel BATCH-lane TTS requests (POST /v1/audio/speech; 1 = serial)
    # max parallel VOICE-lane TTS requests (a live /v1/realtime session's own
    # spoken replies) — a SEPARATE pool from tts_concurrency, not a shared
    # one. See "TTS concurrency lanes" below for why.
    tts_voice_concurrency: int

    # VAD / turn detection (used by the realtime WS pipeline).
    vad_threshold: float
    vad_silence_ms: int
    vad_prefix_padding_ms: int
    vad_max_turn_ms: int
    default_turn_detection: str
    default_aec_mode: str
    barge_in_window_ms: int
    barge_in_model: str | None
    # Stream the generate call and speak it sentence by sentence (approved
    # deviation d7, 2026-09-18). Default TRUE: the whole point is that first
    # audio stops depending on reply length. ``false`` selects the previous,
    # whole-reply path byte-for-byte — the request body loses its ``stream``
    # key and the route drives the non-streaming surface — so the rollback is
    # exact, not approximate.
    generate_stream: bool

    # Where the FastAPI app listens (inside the container).
    host: str
    port: int


# Floor for VAD_MAX_TURN_MS — see build_settings for why a non-positive value
# is not merely useless but actively harmful.
_MIN_MAX_TURN_MS = 1_000

# Floor for TOOL_WAIT_TIMEOUT_MS — same reasoning as _MIN_MAX_TURN_MS: a
# non-positive deadline would force-fail a tool call before it could ever
# complete, defeating tool_use entirely rather than merely being a no-op.
_MIN_TOOL_WAIT_TIMEOUT_MS = 1_000


def _as_int(env: Mapping[str, str], key: str, default: int) -> int:
    try:
        return int(env.get(key) or default)
    except (TypeError, ValueError):
        return int(default)


# Tokens that turn a boolean knob OFF. The DEFAULT-ON direction is what makes
# a denylist right here: a typo (``GENERATE_STREAM=ture``) leaves streaming on
# — the deployed, measured behaviour — rather than silently reverting a
# deployment to the slow path nobody asked for. Mirrors _vocalize.py's own
# truthy-token set, inverted.
_FALSY_TOKENS = frozenset({"0", "false", "no", "off"})


def _is_off(value: str | None) -> bool:
    """True iff *value* explicitly disables a default-on boolean knob."""
    return (value or "").strip().lower() in _FALSY_TOKENS


def _as_float(env: Mapping[str, str], key: str, default: float) -> float:
    try:
        return float(env.get(key) or default)
    except (TypeError, ValueError):
        return float(default)


def _default_system_prompt_for_language(language: str) -> str:
    """The built-in default system prompt for *language* (no operator override).

    ``"he"`` (case-insensitive, region/script subtags ignored — mirrors
    ``_session``'s own shape-only language check) gets
    :data:`DEFAULT_SYSTEM_PROMPT_HE`; every other language, including the
    default ``"en"``, gets the pre-existing
    :data:`_SESSION_DEFAULT_SYSTEM_PROMPT` unchanged.
    """
    primary = (language or "").split("-", 1)[0].lower()
    return DEFAULT_SYSTEM_PROMPT_HE if primary == "he" else _SESSION_DEFAULT_SYSTEM_PROMPT


def build_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Construct :class:`Settings` from environment variables (pure)."""
    env = os.environ if env is None else env
    return Settings(
        tts_url=(env.get("TTS_URL") or "http://chatterbox:9000").rstrip("/"),
        stt_url=(env.get("STT_URL") or "http://stt:9002").rstrip("/"),
        openai_base_url=(env.get("OPENAI_BASE_URL") or "http://gateway:8000").rstrip("/"),
        openai_api_key=env.get("OPENAI_API_KEY") or "EMPTY",
        openai_model=env.get("OPENAI_MODEL") or "",
        language=env.get("REALTIME_LANGUAGE") or _SESSION_DEFAULT_LANGUAGE,
        tool_wait_timeout_ms=max(
            _MIN_TOOL_WAIT_TIMEOUT_MS, _as_int(env, "TOOL_WAIT_TIMEOUT_MS", 120_000)
        ),
        # Empty/unset → the mirrored code-level fallback, same "or default"
        # idiom as every other string field above (an operator who blanks the
        # line in .env gets the safe spoken-style prompt back, not silence).
        # A Hebrew deployment (REALTIME_LANGUAGE=he) gets the Hebrew variant
        # instead of the English one — but only when the operator has NOT
        # set an explicit DEFAULT_SYSTEM_PROMPT; an explicit value always
        # wins outright, in either language.
        default_system_prompt=(
            env.get("DEFAULT_SYSTEM_PROMPT")
            or _default_system_prompt_for_language(
                env.get("REALTIME_LANGUAGE") or _SESSION_DEFAULT_LANGUAGE
            )
        ),
        default_voice=env.get("DEFAULT_VOICE") or "",
        # Clamp to >=1: tts_concurrency seeds an asyncio.Semaphore, and Semaphore(0)
        # (or negative) blocks every TTS request forever; tts_speed is a percentage,
        # where 0/negative would emit nonsensical SSML rate="0%" → backend 502s.
        tts_speed=max(1, _as_int(env, "TTS_SPEED", 125)),
        tts_concurrency=max(1, _as_int(env, "TTS_CONCURRENCY", 1)),
        # Same clamp, same reason (Semaphore(0)/negative blocks forever) — this
        # is the VOICE lane's own independent pool, see "TTS concurrency
        # lanes" below. Defaults to 1, matching tts_concurrency's own
        # historical default: concurrent /v1/realtime sessions are not yet
        # validated (issue #149's follow-up), so this stays conservative
        # rather than assuming untested multi-session headroom — the fix
        # here is ISOLATION from the batch lane, not a throughput increase.
        tts_voice_concurrency=max(1, _as_int(env, "TTS_VOICE_CONCURRENCY", 1)),
        vad_threshold=_as_float(env, "VAD_THRESHOLD", 0.5),
        vad_silence_ms=_as_int(env, "VAD_SILENCE_MS", 600),
        vad_prefix_padding_ms=_as_int(env, "VAD_PREFIX_PADDING_MS", 300),
        # VAD_MAX_TURN_MS: hard cap on one uninterrupted turn before the
        # segmenter force-commits it (lobes.realtime._segmenter's
        # DEFAULT_MAX_TURN_MS=30_000 — same default, now env-tunable). Wired
        # through docker-compose.audio.yml / env.audio.example in #149 t5;
        # this is the settings-side read t5 left for t6 to add.
        # Clamp like tts_speed/tts_concurrency above: the segmenter
        # force-commits once a turn reaches this length, so 0 or a negative
        # value would commit on the FIRST chunk of every turn and keep
        # committing — an event storm plus one STT forward per chunk. One
        # second is the floor a turn could conceivably want.
        vad_max_turn_ms=max(_MIN_MAX_TURN_MS, _as_int(env, "VAD_MAX_TURN_MS", 30_000)),
        default_turn_detection=env.get("DEFAULT_TURN_DETECTION") or "server_vad",
        default_aec_mode=env.get("DEFAULT_AEC_MODE") or "none",
        barge_in_window_ms=_as_int(env, "BARGE_IN_WINDOW_MS", 750),
        barge_in_model=env.get("BARGE_IN_MODEL") or None,
        # Default ON (see the field's own comment): only an explicit falsy
        # token puts the deployment back on the whole-reply path.
        generate_stream=not _is_off(env.get("GENERATE_STREAM")),
        host=env.get("REALTIME_HOST") or "0.0.0.0",  # nosec B104 — bind all inside the container
        port=_as_int(env, "REALTIME_PORT", 8080),
    )


# The container's live settings (env set by the realtime compose service).
settings = build_settings()


# ---------------------------------------------------------------------------
# TTS concurrency lanes (issue #151 t7)
# ---------------------------------------------------------------------------
#
# tts_client.py used to gate every Chatterbox request behind ONE
# module-global asyncio.Semaphore, shared by both the batch
# POST /v1/audio/speech route and a live /v1/realtime session's own spoken
# replies. Once a session can speak (issue #151), that shared gate means a
# voice reply can queue behind unrelated batch TTS work already in flight —
# and in a spoken turn, latency IS dead air.
#
# The fix is two INDEPENDENT semaphore pools, not one raised number: raising
# a shared ceiling only reduces how OFTEN batch traffic blocks a voice
# reply, it cannot guarantee a voice reply never waits behind a batch caller
# that fully saturates the pool (e.g. several long-running batch
# transcriptions). Splitting the pool is a structural guarantee; a bigger
# shared pool is only a probabilistic one. The batch lane's own default
# (tts_concurrency=1) is UNCHANGED — the batch route stays byte-identical.
#
# These helpers build the real semaphore objects (not a duplicate/test-only
# stand-in) — tts_client.py's own lazily-built per-lane registry calls
# straight into new_tts_lane_semaphores(). They live here, in the
# stdlib-only settings module, rather than in tts_client.py, specifically so
# the isolation guarantee is provable by an OFFLINE test:
# asyncio.Semaphore needs no running event loop to construct (true since
# Python 3.10), but tts_client.py imports httpx at module top and is
# therefore excluded from offline coverage (see pyproject.toml's
# [tool.coverage.run] omit list) — no CI lane installs the [realtime] extra.
# See tests/test_realtime_tts_gate.py for the acceptance proof.

BATCH_LANE = "batch"
VOICE_LANE = "voice"


def normalize_tts_lane(lane: str | None) -> str:
    """Map any *lane* value to a known lane, defaulting unknowns to BATCH_LANE.

    An existing caller of ``tts_client.synthesize()`` that passes no lane at
    all (today's every caller) — or a future typo — MUST degrade to the
    long-serving, conservative batch behavior, never silently open a
    brand-new, unbounded pool. Only the exact string ``"voice"`` opts into
    the separate voice-lane gate.
    """
    return VOICE_LANE if lane == VOICE_LANE else BATCH_LANE


def tts_concurrency_for_lane(s: Settings, lane: str | None) -> int:
    """Return the semaphore ceiling *lane* should use, read from *s*."""
    return s.tts_voice_concurrency if normalize_tts_lane(lane) == VOICE_LANE else s.tts_concurrency


def new_tts_lane_semaphores(s: Settings) -> dict[str, asyncio.Semaphore]:
    """Build the two independent TTS concurrency gates, fresh, from *s*.

    Two DISTINCT ``asyncio.Semaphore`` objects — not one shared Semaphore
    sized to the sum of both ceilings — is the point: a batch caller can
    hold every batch-lane permit without touching the voice lane's permits
    at all, and vice versa.
    """
    return {
        BATCH_LANE: asyncio.Semaphore(s.tts_concurrency),
        VOICE_LANE: asyncio.Semaphore(s.tts_voice_concurrency),
    }
