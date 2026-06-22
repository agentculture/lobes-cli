"""Async httpx client for Chatterbox TTS — full-read per sentence.

Sends text to the Chatterbox sidecar (``http://chatterbox:9000/v1/audio/synthesize``)
as plain JSON (no SSML — Chatterbox does not support SSML).  Returns raw PCM16
bytes at 24 kHz mono.

Imports httpx at module top, so it loads only in the ``realtime`` container
(the ``[realtime]`` extra) — never in the base wheel or the gateway.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time

import httpx

from ._settings import settings
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


# Module-level client — reused across requests for connection pooling
_client: httpx.AsyncClient | None = None

# Concurrency gate — limits parallel TTS requests across all sessions
_tts_semaphore: asyncio.Semaphore | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, read=60.0))
    return _client


def _reset_client() -> httpx.AsyncClient:
    """Close the existing client and create a fresh one (stale-connection recovery)."""
    global _client
    if _client is not None and not _client.is_closed:
        log.info("[TTS] resetting HTTP client (stale connection recovery)")
        asyncio.get_event_loop().create_task(_client.aclose())
    _client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, read=60.0))
    return _client


def _get_semaphore() -> asyncio.Semaphore:
    global _tts_semaphore
    if _tts_semaphore is None:
        _tts_semaphore = asyncio.Semaphore(settings.tts_concurrency)
        log.info("[TTS] concurrency gate: max %d parallel requests", settings.tts_concurrency)
    return _tts_semaphore


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
) -> bytes:
    """Synthesize a single chunk of cleaned text via the Chatterbox TTS sidecar.

    Sends a plain JSON POST to the sidecar (no SSML — Chatterbox does not support
    SSML).  The retry loop runs INSIDE the semaphore so that ``_reset_client()``
    cannot race with other requests that share the same ``httpx.AsyncClient``.

    ``speed`` is accepted for API compatibility with callers but is not forwarded
    (Chatterbox has no speed control in the sidecar contract).

    Returns raw PCM16 bytes at 24 kHz (empty on error).
    """
    global _req_counter
    _req_counter += 1
    req_id = _req_counter
    tag = f"[TTS req={req_id}]"

    if cancel_event and cancel_event.is_set():
        return b""

    log.info(
        "%s request: %d chars | %s",
        tag,
        len(clean),
        clean[:120],
    )

    sem = _get_semaphore()
    t_wait = time.monotonic()

    async with sem:
        sem_waited = time.monotonic() - t_wait
        if sem_waited > 0.01:
            log.info("%s semaphore acquired after %.3fs wait", tag, sem_waited)

        for attempt in range(2):  # at most 1 retry
            try:
                client = _get_client()
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
                        _reset_client()
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
                        _reset_client()
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
                    _reset_client()
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
                    _reset_client()
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
) -> bytes:
    """Synthesize text via the Chatterbox TTS sidecar, returning PCM16 audio at 24000Hz.

    Long text is automatically split into chunks.  For the common case (text
    already fits) this returns a single request with no overhead.

    ``speed`` is accepted for API compatibility with callers but is not forwarded
    to Chatterbox (the sidecar has no speed control).

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
        pcm = await _synthesize_single(chunk, url, full_voice, spd, cancel_event)
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
):
    """Compatibility wrapper — calls synthesize() and yields the result as a single chunk."""
    data = await synthesize(
        text, voice=voice, speed=speed, tts_url=tts_url, cancel_event=cancel_event
    )
    if data:
        yield data
