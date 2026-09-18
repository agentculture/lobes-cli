"""Chatterbox Multilingual TTS sidecar — the Hebrew-capable ``chatterbox`` fleet
service (hebrew-realtime plan, task t11; approved deviation d5).

Runs ONLY in the ``chatterbox`` container built from ``Dockerfile.chatterbox-ml``
(the ``[chatterbox]`` extra: fastapi, uvicorn — same as ``chatterbox_server.py``).
The offline dev/CI env has neither those **nor** a GPU, so this module is never
imported by the unit suite for its routes — its routes are thin shells
(``# pragma: no cover``). Every pure helper below (the PCM16 conversion it
reuses from :mod:`lobes.realtime.chatterbox_server`, plus the niqqud/runaway
helpers new to this file) **is** tested directly, offline.

HTTP contract — byte-identical to :mod:`lobes.realtime.chatterbox_server`
(criterion 1: ``tts_client``'s HTTP code needs no change to talk to either)
---------------------------------------------------------------------------
GET  /v1/health/ready            -> 503 ``{"status":"loading"}`` until the model
                                     is loaded AND a warm-up synthesis has
                                     succeeded; 200 ``{"status":"ok"}`` once ready.
POST /v1/audio/synthesize        -> JSON body ``{"text": str, "voice": str|null}``
                                     Response: raw PCM16 mono 24 kHz bytes,
                                     Content-Type: audio/pcm.

What is DIFFERENT from ``chatterbox_server.py`` (both server-side only — the
request/response shape above is unchanged):

1. **Engine**: ``chatterbox.mtl_tts.ChatterboxMultilingualTTS`` (23 languages,
   including Hebrew), not ``chatterbox.tts.ChatterboxTTS``. Its
   ``generate(text, language_id, audio_prompt_path=None, exaggeration=0.5,
   cfg_weight=0.5, ...)`` accepts the same ``audio_prompt_path``/
   ``exaggeration``/``cfg_weight`` keywords as the English class (confirmed by
   reading the installed package source in ``lobes-chatterbox:latest``,
   2026-09-18 — see this task's final report) and returns the same kind of
   float waveform tensor, so :func:`lobes.realtime.chatterbox_server.
   float_tensor_to_pcm16` is reused unchanged rather than re-implemented.
2. **Language is a server setting** (``TTS_LANGUAGE``, default ``"he"`` — see
   :class:`MlSettings`), not a per-request field. Whoever deploys this image
   with ``TTS_LANGUAGE`` UNSET still gets Hebrew, not English: this server
   exists specifically for a non-English deployment (criterion 2), so it
   defaults away from ``chatterbox_server.py``'s implicit English rather than
   mirroring it silently.
3. **Diacritization** (:data:`TTS_DIACRITIZE`, ``auto``/``off``) — see the
   module-level "Diacritization decision table" section below.
4. **A runaway-synthesis guard** — see the "Runaway synthesis guard" section.

Diacritization decision table (``TTS_DIACRITIZE=auto``, the default)
----------------------------------------------------------------------
Chatterbox Multilingual ALSO has its own built-in Hebrew diacritizer
(``chatterbox.models.tokenizers.tokenizer.add_hebrew_diacritics``, which lazily
imports ``dicta_onnx`` and silently no-ops when that package is absent — see
``docs/evidence/2026-09-hebrew-tts-ab-spark.txt``, which measured phonikud's
and dicta's niqqud output as IDENTICAL on the test sentences). Running BOTH
this sidecar's diacritizer (or the upstream bridge's — task t9,
``lobes/realtime/_vocalize.py``) and Chatterbox's own would either double up
harmlessly (same niqqud twice, wasted CPU) or, if the two diacritizers ever
disagree, corrupt the vocalization silently. This server therefore decides
ONE authority for niqqud, deterministically, per request:

    has niqqud?  DIACRITIZE  diacritizer OK?  -> outcome
    -----------  ----------  ---------------  ------------------------------
    yes          auto        (irrelevant)     pass through as-is; Chatterbox's
                                               internal diacritizer bypassed
                                               (monkeypatched at model load —
                                               see _bypass_internal_diacritizer)
    no           auto        yes              sidecar diacritizes via
                                               _vocalize.vocalize_hebrew;
                                               Chatterbox's internal
                                               diacritizer bypassed too (same
                                               monkeypatch — a no-op on
                                               already-vocalized text, but
                                               keeps "one authority"
                                               unconditional in auto mode)
    no           auto        no               synthesized AS-IS
                                               (undiacritized); ONE warning
                                               logged at startup (import/
                                               env-unset) + a per-request
                                               DEBUG log
    (any)        off         (irrelevant)     sidecar never diacritizes;
                                               Chatterbox's internal
                                               diacritizer left ENABLED
                                               (stock behaviour — a no-op
                                               today, since dicta_onnx is not
                                               installed in this image; see
                                               Dockerfile.chatterbox-ml)

    "diacritizer OK?" means phonikud-onnx is importable AND
    ``PHONIKUD_MODEL_PATH`` is set — see :func:`_get_diacritizer`.

Before EITHER the ``has_niqqud`` check or synthesis, incoming text always has
phonikud's invented marks stripped (:func:`strip_phonikud_invented_marks`) —
U+05AB, U+05BD and the literal ``|`` character are phonikud vocabulary, not
standard Unicode niqqud, and ``lobes/realtime/_vocalize.py`` (the upstream
bridge's own diacritization hook, task t9) does NOT strip them before handing
text to ``tts_client.synthesize()`` — confirmed by reading that module in
full; nothing in it mentions U+05AB/U+05BD/``|``. This sidecar is therefore
the one place they are actually removed today, matching the A/B probe's own
methodology (``docs/evidence/2026-09-hebrew-tts-ab-spark.txt``: "phonikud's
invented stress/shva marks (U+05AB, U+05BD, '|') were stripped: standard
niqqud only").

Runaway synthesis guard
------------------------
The A/B measured Chatterbox Multilingual sampling runaway: the SAME 8-word
Hebrew sentence produced 2.92 s of audio (stable) in one arm and 24.5 s (~35
invented words) in another — same text, same engine, different sample
("Chatterbox SAMPLING INSTABILITY, not a diacritizer difference" per the
evidence file). :func:`max_plausible_duration_s` gives a generous per-request
ceiling; :func:`is_runaway` compares a synthesis's actual duration against it.
On a runaway synthesis this server retries (a fresh sample may simply not
run away) up to :data:`MlSettings.max_retries` times, and if it still runs
away, TRUNCATES the PCM16 audio at the last low-energy point before the
ceiling (:func:`find_truncation_sample` / :func:`truncate_pcm16`) rather than
shipping tens of seconds of invented speech or failing the turn outright.
Every retry and every truncation is logged with the measured numbers.
"""

from __future__ import annotations

import logging
import math
import os
import struct
import threading
from collections.abc import Mapping
from dataclasses import dataclass

from ._tts_text import _base_char_length
from ._vocalize import Diacritizer, build_phonikud_diacritizer, vocalize_hebrew
from .chatterbox_server import float_tensor_to_pcm16

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pure stdlib helpers — unit-testable without fastapi/torch/chatterbox
# ---------------------------------------------------------------------------

# Hebrew niqqud/cantillation combining-mark block, U+0591-U+05C7 per Unicode —
# narrowed to U+05B0-U+05C7 here (the vowel-point/dagesh/shin-dot range niqqud
# vocalization actually uses; U+0591-U+05AF is cantillation, which phonikud
# and dicta both also emit as part of "niqqud" output on occasion, but the
# has_niqqud() check only needs to detect that TEXT WAS ALREADY VOCALIZED —
# any mark in this narrower range is sufficient evidence of that).
_NIQQUD_LO = 0x05B0
_NIQQUD_HI = 0x05C7

# Punctuation-like code points inside that range that are NOT vocalization
# marks and must not count as "this text has niqqud":
#   U+05BE MAQAF          (Hebrew hyphen — joins words, like "-")
#   U+05C0 PASEQ           (a vertical-bar word separator)
#   U+05C3 SOF PASUQ       (a Hebrew full-stop-like verse-end mark)
#   U+05C6 NUN HAFUKHA     (an inverted-nun punctuation mark)
_NIQQUD_PUNCTUATION_LIKE = frozenset({0x05BE, 0x05C0, 0x05C3, 0x05C6})

# phonikud's own invented marks (see the module docstring's "Before EITHER…"
# paragraph) — not standard Unicode niqqud, stripped unconditionally before
# has_niqqud()/synthesis. U+05AB (HEBREW ACCENT OLE, a cantillation mark) and
# U+05BD (HEBREW POINT METEG) are real Unicode code points repurposed by
# phonikud for invented stress/shva marks; '|' is a plain ASCII pipe.
_PHONIKUD_INVENTED_MARKS = frozenset({"֫", "ֽ", "|"})


def has_niqqud(text: str) -> bool:
    """True iff *text* contains at least one Hebrew niqqud/vocalization mark.

    Deliberately excludes the punctuation-like code points inside the
    U+05B0-U+05C7 range (see :data:`_NIQQUD_PUNCTUATION_LIKE`) — a maqaf or
    sof-pasuq alone does not mean the text has been vocalized.
    """
    return any(
        _NIQQUD_LO <= ord(ch) <= _NIQQUD_HI and ord(ch) not in _NIQQUD_PUNCTUATION_LIKE
        for ch in text
    )


def strip_phonikud_invented_marks(text: str) -> str:
    """Remove phonikud's invented stress/shva marks and the ``|`` separator.

    Pure, idempotent, and safe to call on text that never had them (returns
    it unchanged). See the module docstring for why this runs unconditionally
    on every request.
    """
    return "".join(ch for ch in text if ch not in _PHONIKUD_INVENTED_MARKS)


# --- Runaway-synthesis guard ------------------------------------------------

# Generous defaults, env-tunable (TTS_MAX_SECONDS_PER_BASE_CHAR /
# TTS_MAX_DURATION_SLACK_S). Checked against the two measured cases in
# docs/evidence/2026-09-hebrew-tts-ab-spark.txt:
#   he1 (8 words, ~43 base chars): stable audio 2.92s (threshold ~11.6s ->
#       passes with margin); runaway audio 24.5s (threshold ~11.6s -> caught,
#       >2x over).
#   he2 (9 words, ~58 base chars): its phonikud-arm 11.56s clip (threshold
#       ~14.6s) also passes — the evidence notes that clip as merely "ran
#       long", not garbage (words were correct), so these defaults do not
#       flag it; a genuinely runaway clip (the Dicta arm's 24.5s on shorter
#       text) is still caught with a wide margin.
_DEFAULT_MAX_SECONDS_PER_BASE_CHAR = 0.20
_DEFAULT_MAX_DURATION_SLACK_S = 3.0

# Truncation search defaults — see find_truncation_sample().
_DEFAULT_TRUNCATE_WINDOW_MS = 20.0
_DEFAULT_TRUNCATE_SILENCE_RATIO = 0.10


def max_plausible_duration_s(
    text: str,
    seconds_per_base_char: float = _DEFAULT_MAX_SECONDS_PER_BASE_CHAR,
    slack_s: float = _DEFAULT_MAX_DURATION_SLACK_S,
) -> float:
    """The longest audio duration that is plausible for *text*.

    Ratio-based over BASE characters (:func:`lobes.realtime._tts_text.
    _base_char_length` — niqqud combining marks never count, mirroring that
    module's own ``_min_plausible_duration``/``_is_truncated`` too-SHORT
    check). Generous on purpose: a false "runaway" flag costs one retry, a
    missed one ships invented speech.
    """
    return _base_char_length(text) * seconds_per_base_char + slack_s


def is_runaway(
    text: str,
    duration_s: float,
    seconds_per_base_char: float = _DEFAULT_MAX_SECONDS_PER_BASE_CHAR,
    slack_s: float = _DEFAULT_MAX_DURATION_SLACK_S,
) -> bool:
    """True when *duration_s* exceeds what is plausible for *text*."""
    return duration_s > max_plausible_duration_s(text, seconds_per_base_char, slack_s)


def pcm16_duration_s(pcm: bytes, sample_rate: int) -> float:
    """Duration, in seconds, of raw PCM16 mono *pcm* at *sample_rate*."""
    if sample_rate <= 0:
        return 0.0
    return (len(pcm) // 2) / sample_rate


def _window_rms(pcm: bytes, start_sample: int, end_sample: int) -> float:
    """RMS amplitude of ``pcm``'s samples in ``[start_sample, end_sample)``."""
    n = end_sample - start_sample
    if n <= 0:
        return 0.0
    samples = struct.unpack(f"<{n}h", pcm[start_sample * 2 : end_sample * 2])
    return math.sqrt(sum(s * s for s in samples) / n)


def find_truncation_sample(
    pcm: bytes,
    sample_rate: int,
    max_samples: int,
    window_ms: float = _DEFAULT_TRUNCATE_WINDOW_MS,
    silence_ratio: float = _DEFAULT_TRUNCATE_SILENCE_RATIO,
) -> int:
    """The sample index to cut *pcm* at: the last low-energy window at or
    before *max_samples*.

    "Low-energy" is relative — a window whose RMS is at most *silence_ratio*
    of the clip's own peak-window RMS. Scans backward from *max_samples* in
    *window_ms* windows; returns *max_samples* itself (a hard cut) if no
    sufficiently quiet window is found, or if the clip is silent throughout
    (peak RMS is 0 — nothing to compare against).
    """
    total_samples = len(pcm) // 2
    max_samples = max(0, min(max_samples, total_samples))
    if max_samples <= 0:
        return 0

    window_samples = max(1, int(sample_rate * window_ms / 1000))

    peak_rms = 0.0
    idx = 0
    while idx < total_samples:
        end = min(idx + window_samples, total_samples)
        r = _window_rms(pcm, idx, end)
        if r > peak_rms:
            peak_rms = r
        idx += window_samples

    if peak_rms <= 0:
        return max_samples

    threshold = peak_rms * silence_ratio
    idx = max_samples
    while idx > 0:
        start = max(0, idx - window_samples)
        r = _window_rms(pcm, start, idx)
        if r <= threshold:
            return idx
        idx -= window_samples

    return max_samples


def truncate_pcm16(
    pcm: bytes,
    sample_rate: int,
    max_duration_s: float,
    window_ms: float = _DEFAULT_TRUNCATE_WINDOW_MS,
    silence_ratio: float = _DEFAULT_TRUNCATE_SILENCE_RATIO,
) -> bytes:
    """Cut *pcm* to at most *max_duration_s*, at the last low-energy point
    found by :func:`find_truncation_sample`. Pure, no side effects."""
    max_samples = int(sample_rate * max_duration_s)
    cut_sample = find_truncation_sample(pcm, sample_rate, max_samples, window_ms, silence_ratio)
    return pcm[: cut_sample * 2]


# ---------------------------------------------------------------------------
# Settings — TTS_LANGUAGE / TTS_DIACRITIZE / TTS_MAX_RETRIES / runaway knobs
# ---------------------------------------------------------------------------


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


def _normalize_diacritize(value: str | None) -> str:
    """``"off"`` (case-insensitive) selects off; every other value (including
    unset/blank/typo'd) selects ``"auto"`` — the safe default degrades toward
    MORE correctness (sidecar-controlled niqqud), never silently toward the
    engine's own unvalidated internal path."""
    return "off" if (value or "").strip().lower() == "off" else "auto"


@dataclass(frozen=True)
class MlSettings:
    """Where this sidecar listens and how it decides Hebrew vocalization."""

    host: str
    port: int
    language: str  # TTS_LANGUAGE — default "he" (criterion 2's server setting)
    diacritize: str  # TTS_DIACRITIZE — "auto" | "off"
    max_retries: int  # TTS_MAX_RETRIES — retries after a runaway synthesis
    phonikud_model_path: str  # PHONIKUD_MODEL_PATH — shared with the bridge (t9)
    max_seconds_per_base_char: float
    max_duration_slack_s: float
    truncate_window_ms: float
    truncate_silence_ratio: float


def build_ml_settings(env: Mapping[str, str] | None = None) -> MlSettings:
    """Construct :class:`MlSettings` from environment variables (pure)."""
    env = os.environ if env is None else env
    return MlSettings(
        host=env.get("CHATTERBOX_HOST") or "0.0.0.0",  # nosec B104 - bind all inside the container
        port=_as_int(env, "CHATTERBOX_PORT", 9000),
        language=env.get("TTS_LANGUAGE") or "he",
        diacritize=_normalize_diacritize(env.get("TTS_DIACRITIZE")),
        max_retries=max(0, _as_int(env, "TTS_MAX_RETRIES", 2)),
        phonikud_model_path=env.get("PHONIKUD_MODEL_PATH") or "",
        max_seconds_per_base_char=max(
            0.01,
            _as_float(env, "TTS_MAX_SECONDS_PER_BASE_CHAR", _DEFAULT_MAX_SECONDS_PER_BASE_CHAR),
        ),
        max_duration_slack_s=max(
            0.0, _as_float(env, "TTS_MAX_DURATION_SLACK_S", _DEFAULT_MAX_DURATION_SLACK_S)
        ),
        truncate_window_ms=max(
            1.0, _as_float(env, "TTS_TRUNCATE_WINDOW_MS", _DEFAULT_TRUNCATE_WINDOW_MS)
        ),
        truncate_silence_ratio=min(
            1.0,
            max(0.0, _as_float(env, "TTS_TRUNCATE_SILENCE_RATIO", _DEFAULT_TRUNCATE_SILENCE_RATIO)),
        ),
    )


def readiness_status(model_loaded: bool, warmup_ok: bool, cuda_poisoned: bool) -> tuple[int, dict]:
    """Return an ``(http_status, body)`` pair reflecting real TTS readiness.

    Pure function — no torch/fastapi/chatterbox imports — unit-testable
    offline. Unlike ``chatterbox_server.readiness_status``, this ALSO gates
    on a successful warm-up synthesis (criterion 4: "readiness must be
    non-200 until the model is loaded AND one short warm-up synthesis has
    succeeded" — cold model load is ~120s, and a loaded-but-never-synthesized
    model has not actually proven it can serve a request).
    """
    if not model_loaded:
        return 503, {"status": "loading"}
    if cuda_poisoned:
        return 503, {"status": "unavailable", "reason": "cuda_context_poisoned"}
    if not warmup_ok:
        return 503, {"status": "loading", "reason": "warmup_pending"}
    return 200, {"status": "ready"}


# ---------------------------------------------------------------------------
# FastAPI sidecar (imported only inside the chatterbox-ml container)
# ---------------------------------------------------------------------------

try:
    import anyio
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, Response

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FASTAPI_AVAILABLE = False

_ml_settings = build_ml_settings()

_model = None  # ChatterboxMultilingualTTS singleton
_model_lock = threading.Lock()
_cuda_poisoned: bool = False
_warmup_ok: bool = False

_diacritizer: Diacritizer | None = None
_diacritizer_attempted = False
_internal_diacritizer_bypassed = False

# Warm-up synthesis text — short, valid in every SUPPORTED_LANGUAGES entry's
# script isn't guaranteed, but for the Hebrew default a short Hebrew greeting
# exercises the real request path (tokenizer, T3, S3Gen) exactly like a real
# request would, unlike a synthetic English string under language_id="he".
_WARMUP_TEXT_HE = "שלום"
_WARMUP_TEXT_FALLBACK = "hello"


def _get_diacritizer() -> Diacritizer | None:  # pragma: no cover
    """Return the process-wide Hebrew diacritizer, building it on first use.

    Mirrors ``tts_client._get_hebrew_diacritizer`` (task t9) but is its own
    module-level singleton — this runs in a separate process/container.
    Returns ``None`` (and logs why, once) when ``PHONIKUD_MODEL_PATH`` is
    unset or the model fails to load.
    """
    global _diacritizer, _diacritizer_attempted
    if _diacritizer_attempted:
        return _diacritizer
    _diacritizer_attempted = True
    if not _ml_settings.phonikud_model_path:
        log.warning(
            "[Chatterbox-ML] TTS_DIACRITIZE=auto but PHONIKUD_MODEL_PATH is unset — "
            "un-vocalized Hebrew text will be synthesized as-is"
        )
        return None
    try:
        _diacritizer = build_phonikud_diacritizer(_ml_settings.phonikud_model_path)
    except Exception:  # noqa: BLE001 - degrade to un-vocalized text, never crash startup
        log.exception(
            "[Chatterbox-ML] failed to build phonikud diacritizer from %s — "
            "un-vocalized Hebrew text will be synthesized as-is",
            _ml_settings.phonikud_model_path,
        )
        _diacritizer = None
    return _diacritizer


def _bypass_internal_diacritizer() -> None:  # pragma: no cover
    """Disable Chatterbox's own ``add_hebrew_diacritics`` call once, guarded.

    Only takes effect in ``auto`` mode (see the module docstring's decision
    table) — ``off`` leaves Chatterbox's internal path enabled (a no-op today
    since this image does not install ``dicta_onnx``, but correct behaviour
    if it ever did). Measured technique: setting
    ``chatterbox.models.tokenizers.tokenizer.add_hebrew_diacritics`` to an
    identity function (docs/evidence/2026-09-hebrew-tts-ab-spark.txt's own
    probe methodology).
    """
    global _internal_diacritizer_bypassed
    if _internal_diacritizer_bypassed or _ml_settings.diacritize != "auto":
        return
    import chatterbox.models.tokenizers.tokenizer as _tokenizer_mod

    _tokenizer_mod.add_hebrew_diacritics = lambda s: s
    _internal_diacritizer_bypassed = True
    log.info(
        "[Chatterbox-ML] bypassed Chatterbox's internal Hebrew diacritizer (TTS_DIACRITIZE=auto)"
    )


def _get_model():  # pragma: no cover
    """Lazy-load the ChatterboxMultilingualTTS model once per process."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # type: ignore[import]

                log.info("[Chatterbox-ML] loading multilingual model (cold-load ~120s) …")
                _bypass_internal_diacritizer()
                _model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
                log.info(
                    "[Chatterbox-ML] model ready — sample rate: %d Hz, language=%s",
                    _model.sr,
                    _ml_settings.language,
                )
    return _model


def _prepare_text_for_synthesis(text: str) -> str:  # pragma: no cover
    """Apply the diacritization decision table to *text*. See module docstring."""
    stripped = strip_phonikud_invented_marks(text)
    if _ml_settings.language.lower() != "he" or _ml_settings.diacritize == "off":
        return stripped
    if has_niqqud(stripped):
        return stripped
    diacritizer = _get_diacritizer()
    if diacritizer is None:
        log.debug("[Chatterbox-ML] synthesizing WITHOUT niqqud (no diacritizer available)")
        return stripped
    return vocalize_hebrew(stripped, diacritizer)


def _generate_once(mdl, text: str, voice: str) -> bytes:  # pragma: no cover
    kwargs: dict = {"exaggeration": 0.5, "cfg_weight": 0.5}
    if voice.lower().endswith(".wav"):
        kwargs["audio_prompt_path"] = voice
    wav_tensor = mdl.generate(text, language_id=_ml_settings.language, **kwargs)
    return float_tensor_to_pcm16(wav_tensor)


def _synthesize_with_runaway_guard(mdl, text: str, voice: str) -> bytes:  # pragma: no cover
    """Generate *text*, retrying/truncating on a runaway synthesis.

    Up to ``TTS_MAX_RETRIES`` retries (a fresh sample may simply not run
    away — the evidence file names this "Chatterbox SAMPLING INSTABILITY").
    Still runaway after every retry -> truncated at the plausible ceiling.
    Every retry/truncation is logged with the measured numbers.
    """
    from .protocol import TTS_SAMPLE_RATE

    pcm = _generate_once(mdl, text, voice)
    for attempt in range(_ml_settings.max_retries + 1):
        duration = pcm16_duration_s(pcm, TTS_SAMPLE_RATE)
        threshold = max_plausible_duration_s(
            text, _ml_settings.max_seconds_per_base_char, _ml_settings.max_duration_slack_s
        )
        if not is_runaway(
            text,
            duration,
            _ml_settings.max_seconds_per_base_char,
            _ml_settings.max_duration_slack_s,
        ):
            return pcm
        if attempt < _ml_settings.max_retries:
            log.warning(
                "[Chatterbox-ML] RUNAWAY synthesis: %.2fs audio > %.2fs plausible ceiling "
                "for %d base chars — retrying (%d/%d)",
                duration,
                threshold,
                _base_char_length(text),
                attempt + 1,
                _ml_settings.max_retries,
            )
            pcm = _generate_once(mdl, text, voice)
            continue
        truncated = truncate_pcm16(
            pcm,
            TTS_SAMPLE_RATE,
            threshold,
            _ml_settings.truncate_window_ms,
            _ml_settings.truncate_silence_ratio,
        )
        log.warning(
            "[Chatterbox-ML] STILL RUNAWAY after %d retries: %.2fs audio > %.2fs ceiling — "
            "TRUNCATED to %.2fs",
            _ml_settings.max_retries,
            duration,
            threshold,
            pcm16_duration_s(truncated, TTS_SAMPLE_RATE),
        )
        return truncated
    return pcm  # pragma: no cover - loop always returns above


if _FASTAPI_AVAILABLE:
    app = FastAPI(title="lobes chatterbox-tts-multilingual", version="1")

    @app.on_event("startup")  # pragma: no cover
    async def _warm_model() -> None:
        def _warm() -> None:
            global _warmup_ok
            try:
                mdl = _get_model()
                warmup_text = (
                    _WARMUP_TEXT_HE
                    if _ml_settings.language.lower() == "he"
                    else _WARMUP_TEXT_FALLBACK
                )
                _generate_once(mdl, _prepare_text_for_synthesis(warmup_text), "")
                _warmup_ok = True
                log.info("[Chatterbox-ML] warm-up synthesis succeeded — ready")
            except Exception:  # noqa: BLE001 - readiness stays 503; never crash startup
                log.exception("[Chatterbox-ML] warm-up synthesis failed — staying not-ready")

        threading.Thread(target=_warm, daemon=True).start()

    @app.get("/v1/health/ready")  # pragma: no cover
    async def health() -> Response:
        code, body = readiness_status(_model is not None, _warmup_ok, _cuda_poisoned)
        return JSONResponse(status_code=code, content=body)

    @app.post("/v1/audio/synthesize")  # pragma: no cover
    async def synthesize(request_body: dict) -> Response:
        """Synthesize text to raw PCM16 mono 24 kHz bytes. See module docstring
        for the diacritization decision table and the runaway guard."""
        global _cuda_poisoned
        text = (request_body.get("text") or "").strip()
        if not text:
            return JSONResponse(
                status_code=400, content={"error": {"message": "text must be non-empty"}}
            )

        voice = request_body.get("voice") or ""

        def _generate() -> bytes:
            mdl = _get_model()
            prepared = _prepare_text_for_synthesis(text)
            return _synthesize_with_runaway_guard(mdl, prepared, voice)

        try:
            pcm = await anyio.to_thread.run_sync(_generate)
        except Exception as exc:
            exc_msg = f"{type(exc).__name__}: {exc}"
            if "cuda" in exc_msg.lower() or "accelerator" in exc_msg.lower():
                _cuda_poisoned = True
                log.warning(
                    "[Chatterbox-ML] CUDA/accelerator error — marking poisoned: %s", exc_msg
                )
                return JSONResponse(
                    status_code=500,
                    content={"error": {"message": "CUDA context error", "detail": exc_msg}},
                )
            raise
        _cuda_poisoned = False
        return Response(content=pcm, media_type="audio/pcm")


def main() -> None:  # pragma: no cover
    """Process entrypoint — ``python -m lobes.realtime.chatterbox_multilingual_server``."""
    logging.basicConfig(level=logging.INFO)
    log.info(
        "starting lobes chatterbox-tts-multilingual on %s:%d (language=%s, diacritize=%s)",
        _ml_settings.host,
        _ml_settings.port,
        _ml_settings.language,
        _ml_settings.diacritize,
    )
    uvicorn.run(app, host=_ml_settings.host, port=_ml_settings.port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    main()
