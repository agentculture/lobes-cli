"""Async httpx client for Chatterbox TTS — full-read per sentence.

Sends text to the Chatterbox sidecar (``http://chatterbox:9000/v1/audio/synthesize``)
as plain JSON (no SSML — Chatterbox does not support SSML).  Returns raw PCM16
bytes at 24 kHz mono.

Imports httpx at module top, so it loads only in the ``realtime`` container
(the ``[realtime]`` extra) — never in the base wheel or the gateway.

**TTS concurrency lanes (issue #151 t7).** ``synthesize()`` takes an
optional ``lane`` — ``"batch"`` (the default, unchanged behavior) for
``POST /v1/audio/speech``, or ``"voice"`` for a live ``/v1/realtime``
session's own spoken reply. The two lanes gate on SEPARATE
``asyncio.Semaphore`` pools (built by
:func:`lobes.realtime._settings.new_tts_lane_semaphores`) so a saturated
batch lane can never make a voice reply queue behind it. Each lane also gets
its OWN ``httpx.AsyncClient`` — not one shared client — because the retry
loop in :func:`_synthesize_single` resets the client it used while still
holding that lane's semaphore, specifically so ``_reset_client()`` cannot
race another request sharing the SAME client; splitting the semaphore
without also splitting the client would silently reopen that exact race
across lanes. See ``lobes/realtime/_settings.py`` for the full rationale.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

import httpx

from ._settings import BATCH_LANE, VOICE_LANE, new_tts_lane_semaphores, normalize_tts_lane, settings

# The pure text helpers (_clean_for_tts, _split_for_tts, trailing_pause_ms,
# _min_plausible_duration, _is_truncated, and the
# _EMOJI_RE/_MARKDOWN_RE/_MAX_CLEAN_CHARS constants) now live in
# lobes.realtime._tts_text — a stdlib-only sibling with no httpx import, so
# they are unit-tested offline (tests/test_realtime_tts_text.py). They are
# imported below under their historical names so every existing caller of
# ``lobes.realtime.tts_client._clean_for_tts`` (etc.) is unaffected; several
# (the regex/constant names) are not called from this module directly, so
# they carry an explicit noqa rather than being silently dropped.
from ._tts_text import (  # noqa: F401 - re-exported for backward compatibility
    _EMOJI_RE,
    _MARKDOWN_RE,
    _MAX_CLEAN_CHARS,
    _clean_for_tts,
    _is_truncated,
    _min_plausible_duration,
    _split_for_tts,
    trailing_pause_ms,
)
from ._vocalize import Diacritizer, LazySingleton, build_phonikud_diacritizer, vocalize_hebrew
from .protocol import TTS_SAMPLE_RATE, resolve_voice

log = logging.getLogger(__name__)

_req_counter = 0  # monotonic request ID for log correlation

# Env var read by _get_hebrew_diacritizer(); a Hebrew-hosting deployment
# points this at its phonikud ONNX checkpoint (docs/specs/2026-09-18-hebrew-realtime.md).
_PHONIKUD_MODEL_PATH_ENV = "PHONIKUD_MODEL_PATH"


# Lazily built, cached at most once per process — building it imports
# phonikud_onnx (see lobes.realtime._vocalize.build_phonikud_diacritizer),
# which only exists inside the realtime container's [hebrew] extra.
#
# Wrapped in a LazySingleton (Qodo finding: "first Hebrew reply freezes all
# sessions") rather than a bare module-global + flag: the old pattern set
# its "attempted" flag BEFORE the (slow) build ran, so a concurrent caller
# could observe "already attempted" and get back `None` mid-build instead of
# waiting for the real result — and the build itself used to run
# synchronously on the event loop, blocking every other session.
# `_get_hebrew_diacritizer()` is safe to call via `asyncio.to_thread` (see
# `_maybe_vocalize_hebrew` below): the lock inside LazySingleton makes
# concurrent callers block on the SAME build rather than each seeing a torn
# intermediate state, and running it off the loop means other coroutines
# keep making progress while it does.
def _build_hebrew_diacritizer() -> Diacritizer | None:
    """The one-shot builder passed to :data:`_hebrew_diacritizer_singleton`.

    Returns ``None`` (and logs why, once) when ``PHONIKUD_MODEL_PATH`` is
    unset or the model fails to load — callers then skip vocalization and
    speak un-vocalized Hebrew rather than fail the whole TTS request.
    """
    model_path = os.environ.get(_PHONIKUD_MODEL_PATH_ENV)
    if not model_path:
        log.warning(
            "[TTS] language=he requested but %s is unset — skipping Hebrew vocalization",
            _PHONIKUD_MODEL_PATH_ENV,
        )
        return None
    try:
        return build_phonikud_diacritizer(model_path)
    # Degrade to un-vocalized text, never crash TTS.
    except Exception:  # noqa: BLE001
        log.exception(
            "[TTS] failed to build phonikud diacritizer from %s=%s — skipping Hebrew vocalization",
            _PHONIKUD_MODEL_PATH_ENV,
            model_path,
        )
        return None


_hebrew_diacritizer_singleton: LazySingleton[Diacritizer] = LazySingleton(_build_hebrew_diacritizer)


def _get_hebrew_diacritizer() -> Diacritizer | None:
    """Return the process-wide Hebrew diacritizer, building it on first use.

    Thread-safe and build-at-most-once via :class:`LazySingleton`. Callers
    that must not block the event loop call this through
    ``asyncio.to_thread`` (see :func:`_maybe_vocalize_hebrew`) rather than
    directly.
    """
    return _hebrew_diacritizer_singleton.get()


# Module-level clients — ONE PER LANE (issue #151 t7), reused across requests
# in that lane for connection pooling. Deliberately not a single shared
# client: _synthesize_single's retry loop resets the client it used while
# still holding that lane's own semaphore, so a reset on one lane can never
# race a request in flight on the OTHER lane's client. See the module
# docstring above for the full rationale.
_clients: dict[str, httpx.AsyncClient | None] = {BATCH_LANE: None, VOICE_LANE: None}

# Concurrency gates — one asyncio.Semaphore per lane, built once from
# _settings.new_tts_lane_semaphores(). "batch" gates POST /v1/audio/speech
# (today's TTS_CONCURRENCY, unchanged); "voice" gates a live /v1/realtime
# session's own spoken replies on a SEPARATE pool (TTS_VOICE_CONCURRENCY) —
# see lobes/realtime/_settings.py for why the two are independent objects.
_tts_semaphores: dict[str, asyncio.Semaphore] | None = None


def _get_client(lane: str) -> httpx.AsyncClient:
    # Normalized on the way in, exactly like _get_semaphore below: `_clients`
    # must only ever be keyed by BATCH_LANE/VOICE_LANE. Keying it by the raw
    # string would let an unknown lane take the batch SEMAPHORE (which
    # normalizes) while opening its own third connection pool — a long-lived
    # httpx.AsyncClient nothing ever closes, and the opposite of the
    # "unknown lane -> batch lane" contract normalize_tts_lane documents.
    lane = normalize_tts_lane(lane)
    client = _clients.get(lane)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, read=60.0))
        _clients[lane] = client
    return client


def _reset_client(lane: str) -> httpx.AsyncClient:
    """Close *lane*'s client and create a fresh one (stale-connection recovery).

    Scoped to *lane* only — the other lane's client, and any request in
    flight on it, is untouched. *lane* is normalized first, for the same
    reason :func:`_get_client` normalizes.
    """
    lane = normalize_tts_lane(lane)
    existing = _clients.get(lane)
    if existing is not None and not existing.is_closed:
        log.info("[TTS] resetting HTTP client for lane=%s (stale connection recovery)", lane)
        asyncio.get_event_loop().create_task(existing.aclose())
    client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, read=60.0))
    _clients[lane] = client
    return client


def _get_semaphore(lane: str) -> asyncio.Semaphore:
    global _tts_semaphores
    if _tts_semaphores is None:
        _tts_semaphores = new_tts_lane_semaphores(settings)
        log.info(
            "[TTS] concurrency gates: batch=%d voice=%d parallel requests",
            settings.tts_concurrency,
            settings.tts_voice_concurrency,
        )
    return _tts_semaphores[normalize_tts_lane(lane)]


class _Retry:
    """Sentinel type: the attempt failed, but a retry within the same semaphore
    hold may still succeed.

    Distinct from ``b""`` (give up) so the two outcomes cannot be confused — an
    empty-bytes "retry" would silently mean "no audio" to the caller. It is a
    dedicated type rather than a bare ``object()`` so the helper below can be
    annotated ``bytes | _Retry``: that keeps the "callers only ever see bytes"
    invariant statically checked instead of asserted by a ``type: ignore``.
    """

    __slots__ = ()


_RETRY = _Retry()


def _handle_tts_response(
    resp: "httpx.Response",
    clean: str,
    tag: str,
    elapsed: float,
    attempt: int,
    lane: str,
) -> bytes | _Retry:
    """Validate one TTS response.

    Returns PCM bytes on success, :data:`_RETRY` when the attempt failed but a
    retry may help, or ``b""`` when the caller should give up and degrade to no
    audio. Split out of :func:`_synthesize_single` to keep that function's
    cognitive complexity inside the gate (Sonar S3776).
    """
    if resp.status_code != 200:
        hdrs = {
            k: v for k, v in resp.headers.items() if k.lower() in ("content-type", "content-length")
        }
        log.error(
            "%s HTTP %d after %.2fs | headers=%s | %s",
            tag,
            resp.status_code,
            elapsed,
            hdrs,
            clean[:80],
        )
        return b""

    pcm_data = resp.content
    if not pcm_data:
        log.error(
            "%s EMPTY response body (0 bytes) after %.2fs | %s",
            tag,
            elapsed,
            clean[:80],
        )
        if attempt == 0:
            log.info("%s resetting client for retry", tag)
            _reset_client(lane)
            return _RETRY
        return b""

    duration = len(pcm_data) / 2 / TTS_SAMPLE_RATE
    log.info(
        "%s result: %d bytes (%.2fs audio) in %.2fs | %s",
        tag,
        len(pcm_data),
        duration,
        elapsed,
        clean[:120],
    )

    if _is_truncated(clean, duration):
        min_expected = _min_plausible_duration(clean)
        if attempt == 0:
            log.warning(
                "%s TRUNCATED: %d chars → %.3fs (expected ≥%.2fs), retrying | %s",
                tag,
                len(clean),
                duration,
                min_expected,
                clean[:80],
            )
            _reset_client(lane)
            return _RETRY  # retry within same semaphore hold
        log.warning(
            "%s STILL TRUNCATED after retry: %d chars → %.3fs (expected ≥%.2fs) | %s",
            tag,
            len(clean),
            duration,
            min_expected,
            clean,
        )

    return pcm_data


def _retry_or_give_up(attempt: int, lane: str) -> bytes | _Retry:
    """After a transient network error on *attempt*, retry once or give up.

    ``attempt`` 0 resets *lane*'s client (stale-connection recovery) and
    signals a retry; any later attempt gives up with ``b""``. Shared by both
    network-error arms of :func:`_attempt_synthesize` so that shared decision
    only needs to be read once.
    """
    if attempt == 0:
        _reset_client(lane)
        return _RETRY
    return b""


async def _attempt_synthesize(
    clean: str,
    url: str,
    voice: str,
    tag: str,
    attempt: int,
    lane: str,
) -> bytes | _Retry:
    """Perform ONE HTTP attempt against the Chatterbox sidecar.

    Returns PCM bytes on success, :data:`_RETRY` when this attempt failed but
    another may help, or ``b""`` when the caller should give up. Split out of
    :func:`_synthesize_single` — together with :func:`_retry_or_give_up` —
    to keep that function's cognitive complexity inside the gate (Sonar
    S3776); the try/except tree that used to live inline is unchanged, just
    relocated.
    """
    try:
        client = _get_client(lane)
        t0 = time.monotonic()
        resp = await client.post(
            url,
            json={"text": clean, "voice": voice},
        )
        elapsed = time.monotonic() - t0
        return _handle_tts_response(resp, clean, tag, elapsed, attempt, lane)

    except httpx.ConnectError:
        log.error("%s connect error to %s (attempt %d)", tag, url, attempt + 1)
        return _retry_or_give_up(attempt, lane)
    except httpx.ReadTimeout:
        log.error(
            "%s read timeout after %.0fs (attempt %d) | %s",
            tag,
            time.monotonic() - t0,
            attempt + 1,
            clean[:80],
        )
        return _retry_or_give_up(attempt, lane)
    except Exception as e:  # noqa: BLE001 - log and fail soft; caller degrades to no audio
        # log.exception (not log.error): this is the catch-all arm, so the
        # traceback is the only thing that says WHICH unexpected failure hit.
        log.exception("%s error (%s, attempt %d): %s", tag, type(e).__name__, attempt + 1, e)
        return b""


async def _synthesize_single(
    clean: str,
    url: str,
    voice: str,
    speed: int,
    cancel_event: asyncio.Event | None = None,
    lane: str = BATCH_LANE,
) -> bytes:
    """Synthesize a single chunk of cleaned text via the Chatterbox TTS sidecar.

    Sends a plain JSON POST to the sidecar (no SSML — Chatterbox does not support
    SSML).  The retry loop runs INSIDE the semaphore so that ``_reset_client()``
    cannot race with other requests that share the same ``httpx.AsyncClient`` —
    *lane* pins BOTH the semaphore and the client to the same pool (issue
    #151 t7), so that guarantee holds within a lane and a reset on one lane
    can never race a request on the other.

    ``speed`` is accepted for API compatibility with callers but is not forwarded
    (Chatterbox has no speed control in the sidecar contract).

    ``lane`` defaults to ``BATCH_LANE`` — an existing caller that passes
    nothing behaves exactly as before this task. Pass ``VOICE_LANE`` for a
    live ``/v1/realtime`` session's own spoken reply so it never queues
    behind unrelated batch TTS work.

    Returns raw PCM16 bytes at 24 kHz (empty on error).
    """
    # Normalize once, up front, so the log tag names the lane actually used.
    # Tagging the raw value would print `lane=voise` on a request served by the
    # batch pool — the one place this module talks to a human, lying.
    lane = normalize_tts_lane(lane)
    global _req_counter
    _req_counter += 1
    req_id = _req_counter
    tag = f"[TTS req={req_id} lane={lane}]"

    if cancel_event and cancel_event.is_set():
        return b""

    log.info(
        "%s request: %d chars | %s",
        tag,
        len(clean),
        clean[:120],
    )

    sem = _get_semaphore(lane)
    t_wait = time.monotonic()

    async with sem:
        sem_waited = time.monotonic() - t_wait
        if sem_waited > 0.01:
            log.info("%s semaphore acquired after %.3fs wait", tag, sem_waited)

        for attempt in range(2):  # at most 1 retry
            outcome = await _attempt_synthesize(clean, url, voice, tag, attempt, lane)
            # isinstance, not `is _RETRY`: identity against a module global does
            # not narrow the union for a type checker, so the bare `return` below
            # would still need a suppression.
            if isinstance(outcome, _Retry):
                continue
            return outcome

        return b""  # should not reach here


async def _maybe_vocalize_hebrew(
    clean: str,
    language: str,
    timings_out: dict | None,
) -> str:
    """Run Hebrew vocalization on *clean* when *language* is ``"he"``,
    entirely off the event loop, returning *clean* unchanged otherwise.

    Extracted out of :func:`synthesize` to keep that function's cognitive
    complexity inside the gate (Sonar S3776); behavior and log lines are
    unchanged. Both steps run via ``asyncio.to_thread``:

    - ``_get_hebrew_diacritizer()`` (Qodo finding): it lazily BUILDS the
      phonikud ONNX model on first use, which is slow — calling it directly
      on the event loop froze every other concurrent session for the
      duration of that first build. ``LazySingleton`` (see
      ``lobes.realtime._vocalize``) keeps the build itself safe under
      concurrent callers.
    - :func:`lobes.realtime._vocalize.vocalize_hebrew` — unchanged from
      before this task, already run off the loop with its own timeout.

    When the diacritizer is unavailable (env unset, or it failed to load),
    this degrades to returning *clean* un-vocalized, exactly as before.
    """
    if language != "he":
        return clean

    diacritizer = await asyncio.to_thread(_get_hebrew_diacritizer)
    if diacritizer is None:
        return clean

    started = time.monotonic()
    vocalized = await asyncio.to_thread(vocalize_hebrew, clean, diacritizer)
    if timings_out is not None:
        timings_out["phonikud"] = int((time.monotonic() - started) * 1000)
    return vocalized


async def synthesize(
    text: str,
    voice: str | None = None,
    speed: int | None = None,
    tts_url: str | None = None,
    cancel_event: asyncio.Event | None = None,
    lane: str = BATCH_LANE,
    language: str = "en",
    timings_out: dict | None = None,
) -> bytes:
    """Synthesize text via the Chatterbox TTS sidecar, returning PCM16 audio at 24000Hz.

    Long text is automatically split into chunks.  For the common case (text
    already fits) this returns a single request with no overhead.

    ``speed`` is accepted for API compatibility with callers but is not forwarded
    to Chatterbox (the sidecar has no speed control).

    ``lane`` (issue #151 t7) selects which concurrency pool gates this call —
    ``BATCH_LANE`` (the default) for the batch ``POST /v1/audio/speech``
    route, unchanged from before this task, or ``VOICE_LANE`` for a live
    ``/v1/realtime`` session's own spoken reply, on its own SEPARATE pool so
    it never queues behind unrelated batch TTS work. An existing caller that
    passes nothing gets exactly today's behavior.

    ``language`` (issue hebrew-realtime t9) defaults to ``"en"`` — cleaned/
    split output for that default is byte-identical to before this task. A
    caller (wired by a later task) passes ``"he"`` to run the cleaned text
    through :func:`lobes.realtime._vocalize.vocalize_hebrew` — using the
    process-wide diacritizer built from ``PHONIKUD_MODEL_PATH`` — after
    cleaning and before chunking, so niqqud reaches both the chunk-size
    accounting (which counts base characters, not niqqud combining marks —
    see ``lobes.realtime._tts_text._base_char_length``) and the sidecar
    itself. When the diacritizer is unavailable (env unset, or it failed to
    load), the request degrades to un-vocalized Hebrew rather than failing.

    ``timings_out`` (hebrew-realtime t12) is an optional mapping this call
    writes its own measured stages into — today exactly one, ``"phonikud"``:
    the milliseconds spent in the diacritizer, which happens INSIDE this
    function and is therefore unobservable to the route that reports it on
    ``response.done``. Written only when vocalization actually ran, so an
    English reply leaves the mapping untouched and every existing caller
    (which passes none at all) is unaffected.

    Returns:
        Raw PCM16 bytes at 24000Hz (empty bytes if nothing to synthesize).
    """
    url = (tts_url or settings.tts_url).rstrip("/") + "/v1/audio/synthesize"
    full_voice = resolve_voice(voice or settings.default_voice)
    spd = speed if speed is not None else settings.tts_speed

    if speed is not None and speed != 100:
        log.warning("[TTS] speed=%d requested but Chatterbox has no speed control — ignored", speed)

    # Clean text: strip emoji, markdown, normalize whitespace
    clean = _clean_for_tts(text)
    if not clean:
        log.debug("[TTS] skipping empty text after cleanup (original: %s)", text[:40])
        return b""

    clean = await _maybe_vocalize_hebrew(clean, language, timings_out)

    # Split into chunks that fit within the conservative Chatterbox ceiling
    chunks = _split_for_tts(clean)
    if len(chunks) > 1:
        log.warning("[TTS] text too long (%d chars), split into %d chunks", len(clean), len(chunks))

    pcm_parts: list[bytes] = []
    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            log.info("[TTS] chunk %d/%d (%d chars)", i + 1, len(chunks), len(chunk))
        pcm = await _synthesize_single(chunk, url, full_voice, spd, cancel_event, lane=lane)
        if pcm:
            pcm_parts.append(pcm)
    return b"".join(pcm_parts)


# Keep backward-compat alias for any callers using the streaming API
async def synthesize_stream(
    text: str,
    voice: str | None = None,
    speed: int | None = None,
    tts_url: str | None = None,
    cancel_event: asyncio.Event | None = None,
    lane: str = BATCH_LANE,
    language: str = "en",
    timings_out: dict | None = None,
):
    """Compatibility wrapper — calls synthesize() and yields the result as a single chunk."""
    data = await synthesize(
        text,
        voice=voice,
        speed=speed,
        tts_url=tts_url,
        cancel_event=cancel_event,
        lane=lane,
        language=language,
        timings_out=timings_out,
    )
    if data:
        yield data
