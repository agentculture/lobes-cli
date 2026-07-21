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
import re
import time

import httpx

from ._settings import BATCH_LANE, VOICE_LANE, new_tts_lane_semaphores, normalize_tts_lane, settings
from .protocol import TTS_SAMPLE_RATE, resolve_voice

log = logging.getLogger(__name__)

_req_counter = 0  # monotonic request ID for log correlation

# Regex to strip emoji (Supplementary Multilingual Plane + common emoji ranges)
_EMOJI_RE = re.compile(
    "[\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map
    "\U0001f1e0-\U0001f1ff"  # flags
    "\U00002702-\U000027b0"  # dingbats
    "\U0000fe00-\U0000fe0f"  # variation selectors
    "\U0000200d"  # zero-width joiner
    "\U000024c2-\U0001f251"
    "]+",
    flags=re.UNICODE,
)

# Markdown-style formatting
_MARKDOWN_RE = re.compile(r"[*_~`#]")

# Max chars of *cleaned* text per TTS request.
# Conservative chunking ceiling for Chatterbox (no hard SSML or Triton token limit;
# kept at 600 to avoid extremely long single requests and preserve latency).
_MAX_CLEAN_CHARS = 600


def _split_for_tts(text: str, max_chars: int = _MAX_CLEAN_CHARS) -> list[str]:
    """Split *text* into chunks of at most *max_chars* characters.

    Tries to break at the last ``", "`` before the limit, then last ``" "``,
    and hard-cuts only as a last resort.  Returns a single-element list when
    the text already fits.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        # Prefer splitting at last ", " (natural pause)
        idx = window.rfind(", ")
        if idx > 0:
            cut = idx + 2  # keep the comma+space with the left chunk
        else:
            # Fall back to last space
            idx = window.rfind(" ")
            if idx > 0:
                cut = idx + 1
            else:
                # Hard cut — no good break point
                cut = max_chars
        chunk = remaining[:cut].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[cut:].strip()

    if remaining:
        chunks.append(remaining)
    return chunks


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


def _clean_for_tts(text: str) -> str:
    """Strip emoji, markdown, dashes, quotes and normalize for TTS input."""
    text = _EMOJI_RE.sub(" ", text)
    text = _MARKDOWN_RE.sub("", text)
    # Em-dash / en-dash → comma (natural pause; raw dashes confuse TTS)
    text = text.replace("—", ", ")
    text = text.replace("–", ", ")
    # Curly single quotes / apostrophes → ASCII apostrophe (preserves contractions)
    text = text.replace("‘", "'")
    text = text.replace("’", "'")
    # Strip double-quotes (TTS doesn't need to voice them)
    text = re.sub(r'["“”]', "", text)
    # Remove markdown list markers at line start:  - item  /  1. item
    text = re.sub(r"(?m)^\s*-\s+", " ", text)
    text = re.sub(r"(?m)^\s*\d+[.)]\s+", " ", text)
    # Collapse whitespace / newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Punctuation-aware pause helpers
# ---------------------------------------------------------------------------


def trailing_pause_ms(original_text: str) -> int:
    """Return inter-sentence silence duration (ms) based on ending punctuation.

    Examines the *original* sentence text (before TTS cleaning) so that
    trailing emoji and raw punctuation are still visible.
    """
    s = original_text.rstrip()
    if not s:
        return 200

    # Check multi-char patterns first (longest match wins)
    if re.search(r"!{3,}$", s):
        return 400
    if s.endswith("?!") or s.endswith("!?"):
        return 350
    if s.endswith("!!"):
        return 350
    if s.endswith("...") or s.endswith("…"):
        return 400
    if s.endswith("."):
        return 350
    if s.endswith("?"):
        return 350
    if s.endswith("!"):
        return 300

    # Trailing emoji
    if _EMOJI_RE.search(s[-2:]):
        return 250

    return 200


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
            try:
                client = _get_client(lane)
                t0 = time.monotonic()
                resp = await client.post(
                    url,
                    json={"text": clean, "voice": voice},
                )
                elapsed = time.monotonic() - t0

                if resp.status_code != 200:
                    hdrs = {
                        k: v
                        for k, v in resp.headers.items()
                        if k.lower() in ("content-type", "content-length")
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
                        continue
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

                # Detect truncated audio — ratio-based: expect at least 15ms per char
                # (normal speech at 125% ≈ 60-80ms/char; 15ms is very conservative)
                min_expected = max(0.5, len(clean) * 0.015)
                if len(clean) > 10 and duration < min_expected:
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
                        continue  # retry within same semaphore hold
                    else:
                        log.warning(
                            "%s STILL TRUNCATED after retry: %d chars → %.3fs "
                            "(expected ≥%.2fs) | %s",
                            tag,
                            len(clean),
                            duration,
                            min_expected,
                            clean,
                        )

                return pcm_data

            except httpx.ConnectError:
                log.error("%s connect error to %s (attempt %d)", tag, url, attempt + 1)
                if attempt == 0:
                    _reset_client(lane)
                    continue
                return b""
            except httpx.ReadTimeout:
                log.error(
                    "%s read timeout after %.0fs (attempt %d) | %s",
                    tag,
                    time.monotonic() - t0,
                    attempt + 1,
                    clean[:80],
                )
                if attempt == 0:
                    _reset_client(lane)
                    continue
                return b""
            except (
                Exception
            ) as e:  # noqa: BLE001 - log and fail soft; the caller degrades to no audio
                log.error("%s error (%s, attempt %d): %s", tag, type(e).__name__, attempt + 1, e)
                return b""

        return b""  # should not reach here


async def synthesize(
    text: str,
    voice: str | None = None,
    speed: int | None = None,
    tts_url: str | None = None,
    cancel_event: asyncio.Event | None = None,
    lane: str = BATCH_LANE,
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
):
    """Compatibility wrapper — calls synthesize() and yields the result as a single chunk."""
    data = await synthesize(
        text,
        voice=voice,
        speed=speed,
        tts_url=tts_url,
        cancel_event=cancel_event,
        lane=lane,
    )
    if data:
        yield data
