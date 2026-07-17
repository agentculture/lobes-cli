"""Per-role RUNTIME measurement — issue #81, task t8.

Probes each of the seven first-class roles (:data:`lobes.roles.ROLES`) on its own
live endpoint and reports **runtime/serving** metrics, organised BY ROLE and
grouped by the metric family its ``runtime`` implies:

* **LLM roles** (``cortex``, ``senses``): TTFT, decode throughput, prefill
  throughput, served context, and — when the vLLM ``/metrics`` scrape is
  cheaply reachable — KV-cache memory usage. Restart/error counts are not
  cheaply available without a docker inspect (forbidden here — see below), so
  they are always reported ``null`` rather than invented.
* **embedder / reranker** (pooling): requests/sec, docs/sec, latency, batch
  size, loaded state.
* **stt / tts** (audio overlay sidecars): input/output duration, latency,
  real-time factor (RTF), failure rate.

RUNTIME-ONLY — boundary c7/h14
-------------------------------
Every field this module ever emits is a serving/runtime measurement: latency,
throughput, RTF, memory, readiness, failure rate. **No field here may assert
answer correctness, task quality, or agent-task success** — judging whether an
*answer* was any good is Colleague's job, not lobes'. :data:`ALLOWED_METRIC_KEYS`
is the closed vocabulary this module is allowed to emit; a reviewer (or a test)
can diff any new key against it to catch a boundary violation before it ships.

Read-only
---------
Every probe here is a plain HTTP GET/POST against a role's own endpoint (health
check, a tiny generation, a tiny embedding/rerank batch, a tiny synthetic audio
round trip). Nothing in this module imports :mod:`lobes.runtime._compose` or
shells out to docker/compose, and nothing here ever raises on a network failure
— nothing calls this module needs to catch anything.  A role that is not
``loaded`` (see :mod:`lobes.roles`) or whose endpoint refuses the connection is
reported with ``ready=False`` and every measured metric ``null`` — this is the
*normal* case in CI/tests (no live models) and on an unscaffolded deployment,
not an error.

Every probe uses a short timeout (:data:`DEFAULT_TIMEOUT`) so an unreachable
endpoint fails fast instead of hanging the CLI.

Reuse
-----
The LLM-role probes reuse :mod:`lobes.assess`'s stdlib ``urllib`` primitives
directly: :func:`lobes.assess.measure_prefill_ttft` (TTFT — and, since a
``max_tokens=1`` request is prefill-dominated, ``prompt_tokens / ttft`` doubles
as the prefill-throughput estimate) and :func:`lobes.assess._post` (the same
POST-and-time helper :func:`lobes.assess.run_benchmark` uses for decode
throughput). ``_post`` gained an optional ``path`` keyword (default unchanged)
so the JSON-in/JSON-out pooling roles (embedder → ``/v1/embeddings``, reranker
→ ``/v1/rerank``) reuse it too. The audio roles (stt/tts) need a raw-bytes
response (audio) and a multipart request (the STT upload) — genuinely
different wire shapes ``_post`` can't express — so those two write their own
minimal ``urllib`` calls, timed the same way (``time.monotonic()`` around
``urlopen``).
"""

from __future__ import annotations

import io
import json
import time
import urllib.request
import uuid
import wave

from lobes import _metrics
from lobes import assess as _assess
from lobes.roles import ROLES, RoleInfo

# Short probe timeout (seconds): graceful degradation depends on this being
# small — an unreachable role must fail fast, not hang the CLI.
DEFAULT_TIMEOUT = 5.0

# ---------------------------------------------------------------------------
# The RUNTIME-ONLY metric vocabulary (boundary c7/h14). Every key below is a
# serving/runtime measurement; none may assert correctness/quality/success.
# ---------------------------------------------------------------------------

LLM_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "ttft_ms",  # time-to-first-token
        "decode_tps",  # decode throughput, tokens/sec
        "prefill_tps",  # prefill throughput, tokens/sec
        "context",  # served --max-model-len (always known — from the registry)
        "mem_usage_pct",  # vLLM KV-cache usage 0..1, when /metrics is reachable
        "restart_count",  # not cheaply available without docker inspect — null
        "error_count",  # not cheaply available from /metrics alone — null
    }
)

EMBED_RERANK_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "reqs_per_sec",
        "docs_per_sec",
        "latency_ms",
        "batch_size",
        "loaded",
    }
)

AUDIO_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "rtf",  # real-time factor: processing_time / audio_duration
        "latency_ms",
        "duration_ms",  # audio duration — input (stt) or output (tts)
        "failure_rate",
    }
)

ALLOWED_METRIC_KEYS: frozenset[str] = LLM_METRIC_KEYS | EMBED_RERANK_METRIC_KEYS | AUDIO_METRIC_KEYS

_LLM_ROLES: tuple[str, ...] = ("cortex", "senses", "muse")
_EMBED_RERANK_ROLES: tuple[str, ...] = ("embedder", "reranker")
_AUDIO_ROLES: tuple[str, ...] = ("stt", "tts")

_FAMILY_BY_ROLE: dict[str, str] = {
    "cortex": "llm",
    "senses": "llm",
    "muse": "llm",
    "embedder": "embed_rerank",
    "reranker": "embed_rerank",
    "stt": "audio",
    "tts": "audio",
}

_EMBED_RERANK_PATH: dict[str, str] = {"embedder": "/v1/embeddings", "reranker": "/v1/rerank"}

# Small fixed probe payloads — cheap enough to run on every `lobes measure`
# call, not a load test (see lobes benchmark / lobes assess for those).
_EMBED_PROBE_INPUT = ["The quick brown fox jumps over the lazy dog."] * 4
_RERANK_PROBE_QUERY = "example query for a runtime latency probe"
_RERANK_PROBE_DOCS = ["document one", "document two", "document three", "document four"]
_DECODE_PROBE_TOKENS = 24
_TTFT_PROBE_INPUT_LEN = 512
_TTS_PROBE_TEXT = "This is a runtime latency probe."
_STT_PROBE_DURATION_S = 0.5
_STT_PROBE_RATE_HZ = 16000


def _empty_metrics(keys: frozenset[str]) -> dict[str, object]:
    return dict.fromkeys(keys)


# ---------------------------------------------------------------------------
# LLM roles — cortex / senses
# ---------------------------------------------------------------------------


def _measure_llm_role(info: RoleInfo, *, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """RUNTIME probe for an LLM role: ttft/decode/prefill/context/mem.

    Never raises. ``context`` is always populated (known from the role
    registry without a probe); everything else stays ``null`` when the role
    isn't loaded, its endpoint refuses the connection, or a probe mid-flight
    fails.
    """
    metrics = _empty_metrics(LLM_METRIC_KEYS)
    metrics["context"] = info.context
    if not info.loaded or not info.endpoint:
        return {"ready": False, "metrics": metrics}

    probe = _metrics.probe_backend(info.endpoint, timeout=timeout)
    if probe["health"] != "ok":
        return {"ready": False, "metrics": metrics}
    live = probe.get("metrics") or {}
    metrics["mem_usage_pct"] = live.get("kv_cache_usage")

    try:
        ttft = _assess.measure_prefill_ttft(
            info.endpoint,
            info.model,
            input_len=_TTFT_PROBE_INPUT_LEN,
            timeout=max(1, int(timeout)),
        )
        metrics["ttft_ms"] = ttft["ttft_ms"]
        if ttft["ttft_ms"]:
            metrics["prefill_tps"] = round(ttft["prompt_tokens"] / (ttft["ttft_ms"] / 1000.0), 1)

        t0 = time.monotonic()
        decoded = _assess._post(
            info.endpoint,
            {
                "model": info.model,
                "messages": [{"role": "user", "content": "Write a short paragraph."}],
                "max_tokens": _DECODE_PROBE_TOKENS,
                "temperature": 0,
                "ignore_eos": True,
            },
            timeout=timeout,
        )
        dt = time.monotonic() - t0
        completion_tokens = decoded["usage"]["completion_tokens"]
        metrics["decode_tps"] = round(completion_tokens / dt, 1) if dt > 0 else None
    except (OSError, KeyError, IndexError, TypeError, ValueError):
        # A probe that started succeeding (health ok) but failed mid-flight is
        # still "not ready" — the mem_usage_pct already gathered above is kept
        # (it's a fact we do have), everything else stays null.
        return {"ready": False, "metrics": metrics}

    return {"ready": True, "metrics": metrics}


# ---------------------------------------------------------------------------
# embedder / reranker (pooling)
# ---------------------------------------------------------------------------


def _embed_rerank_payload(role: str, model: str) -> tuple[dict, int]:
    if role == "embedder":
        payload = {"model": model, "input": list(_EMBED_PROBE_INPUT)}
        return payload, len(_EMBED_PROBE_INPUT)
    payload = {"model": model, "query": _RERANK_PROBE_QUERY, "documents": list(_RERANK_PROBE_DOCS)}
    return payload, len(_RERANK_PROBE_DOCS)


def _measure_embed_rerank_role(info: RoleInfo, *, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """RUNTIME probe for a pooling role: reqs/sec, docs/sec, latency, batch, loaded.

    Never raises — degrades to ``ready=False`` / null metrics on an unloaded
    role, an unreachable endpoint, or a mid-flight failure.
    """
    metrics = _empty_metrics(EMBED_RERANK_METRIC_KEYS)
    metrics["loaded"] = info.loaded
    if not info.loaded or not info.endpoint:
        return {"ready": False, "metrics": metrics}
    if not _metrics.health_ok(info.endpoint, timeout=timeout):
        return {"ready": False, "metrics": metrics}

    payload, batch_size = _embed_rerank_payload(info.role, info.model)
    path = _EMBED_RERANK_PATH[info.role]
    try:
        t0 = time.monotonic()
        _assess._post(info.endpoint, payload, timeout=timeout, path=path)
        dt = time.monotonic() - t0
    except (OSError, KeyError, ValueError, TypeError):
        return {"ready": False, "metrics": metrics}

    metrics["latency_ms"] = round(dt * 1000, 1)
    metrics["reqs_per_sec"] = round(1 / dt, 3) if dt > 0 else None
    metrics["docs_per_sec"] = round(batch_size / dt, 3) if dt > 0 else None
    metrics["batch_size"] = batch_size
    return {"ready": True, "metrics": metrics}


# ---------------------------------------------------------------------------
# stt / tts (audio overlay sidecars, via the realtime facade)
# ---------------------------------------------------------------------------


def _wav_duration_seconds(body: bytes) -> float | None:
    """Best-effort WAV duration from a RIFF/WAVE body; ``None`` on any parse failure."""
    try:
        with wave.open(io.BytesIO(body), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / rate if rate else None
    except (wave.Error, EOFError):
        return None


def _measure_tts_role(info: RoleInfo, *, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """RUNTIME probe for tts: synthesize a short phrase, time it, derive RTF.

    Never raises. The response is raw audio (not JSON), so this can't reuse
    ``assess._post`` (which always ``json.load``s the body) — it POSTs the
    same OpenAI ``/v1/audio/speech`` JSON request directly and times the
    round trip the same way (``time.monotonic()``).
    """
    metrics = _empty_metrics(AUDIO_METRIC_KEYS)
    if not info.loaded or not info.endpoint:
        return {"ready": False, "metrics": metrics}
    if not _metrics.health_ok(info.endpoint, timeout=timeout):
        metrics["failure_rate"] = 1.0
        return {"ready": False, "metrics": metrics}

    payload = {"model": info.model, "input": _TTS_PROBE_TEXT, "response_format": "wav"}
    req = urllib.request.Request(
        info.endpoint.rstrip("/") + "/v1/audio/speech",
        data=json.dumps(payload).encode(),
        # The audio probes build their own Request (raw-audio response — can't
        # reuse assess._post), so they merge the SAME contextvar-scoped auth
        # header assess's primitives attach (#129 items 1-2): with inbound
        # gateway auth on, the probe authenticates like every other verb; with
        # no key installed the dict is empty and the request is byte-identical.
        headers={"Content-Type": "application/json", **_assess._current_auth_headers()},
    )
    try:
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=timeout) as r:  # audio overlay endpoint only
            body = r.read()
        dt = time.monotonic() - t0
    except OSError:
        metrics["failure_rate"] = 1.0
        return {"ready": False, "metrics": metrics}

    metrics["latency_ms"] = round(dt * 1000, 1)
    metrics["failure_rate"] = 0.0
    duration_s = _wav_duration_seconds(body)
    if duration_s:
        metrics["duration_ms"] = round(duration_s * 1000, 1)
        metrics["rtf"] = round(dt / duration_s, 4)
    return {"ready": True, "metrics": metrics}


def _synthetic_probe_wav(
    duration_s: float = _STT_PROBE_DURATION_S, rate: int = _STT_PROBE_RATE_HZ
) -> bytes:
    """A tiny synthetic silent WAV — cheap input for the stt RTF probe.

    Correctness of the transcript is irrelevant (RUNTIME-ONLY): only the round
    trip is timed, against a KNOWN input duration.
    """
    n_frames = int(duration_s * rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return buf.getvalue()


def _multipart_encode(
    fields: dict[str, str], file_field: str, filename: str, content: bytes, content_type: str
) -> tuple[bytes, str]:
    """Minimal ``multipart/form-data`` encoder (stdlib has none) for the STT upload."""
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        (
            f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode()
    )
    parts.append(content)
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _measure_stt_role(info: RoleInfo, *, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """RUNTIME probe for stt: transcribe a tiny synthetic clip, time it, derive RTF.

    Never raises. Needs a multipart upload — a wire shape ``assess._post``
    can't express (JSON-only) — so this builds its own minimal
    ``multipart/form-data`` body and times the round trip the same way.
    """
    metrics = _empty_metrics(AUDIO_METRIC_KEYS)
    if not info.loaded or not info.endpoint:
        return {"ready": False, "metrics": metrics}
    if not _metrics.health_ok(info.endpoint, timeout=timeout):
        metrics["failure_rate"] = 1.0
        return {"ready": False, "metrics": metrics}

    body, content_type = _multipart_encode(
        {"language": "en"}, "file", "probe.wav", _synthetic_probe_wav(), "audio/wav"
    )
    req = urllib.request.Request(
        info.endpoint.rstrip("/") + "/v1/audio/transcriptions",
        data=body,
        # Same contextvar-scoped auth as the tts probe above (#129 items 1-2).
        headers={
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
            **_assess._current_auth_headers(),
        },
    )
    try:
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=timeout) as r:  # audio overlay endpoint only
            r.read()
        dt = time.monotonic() - t0
    except OSError:
        metrics["failure_rate"] = 1.0
        return {"ready": False, "metrics": metrics}

    metrics["latency_ms"] = round(dt * 1000, 1)
    metrics["duration_ms"] = round(_STT_PROBE_DURATION_S * 1000, 1)
    metrics["rtf"] = round(dt / _STT_PROBE_DURATION_S, 4)
    metrics["failure_rate"] = 0.0
    return {"ready": True, "metrics": metrics}


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_MEASURE_FN = {
    "cortex": _measure_llm_role,
    "senses": _measure_llm_role,
    "muse": _measure_llm_role,
    "embedder": _measure_embed_rerank_role,
    "reranker": _measure_embed_rerank_role,
    "stt": _measure_stt_role,
    "tts": _measure_tts_role,
}


def measure_role(role: str, info: RoleInfo, *, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """RUNTIME measurement for one role — dispatches by its metric family.

    Never raises (every family function degrades gracefully — see above).
    Returns::

        {"role", "family", "model", "runtime", "endpoint", "loaded", "ready", "metrics"}

    ``metrics`` carries only keys from that role's family vocabulary (a subset
    of :data:`ALLOWED_METRIC_KEYS`) — never a correctness/quality field.
    """
    result = _MEASURE_FN[role](info, timeout=timeout)
    return {
        "role": role,
        "family": _FAMILY_BY_ROLE[role],
        "model": info.model,
        "runtime": info.runtime,
        "endpoint": info.endpoint,
        "loaded": info.loaded,
        "ready": result["ready"],
        "metrics": result["metrics"],
    }


def measure_registry(
    registry: dict[str, RoleInfo],
    *,
    roles: tuple[str, ...] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, dict]:
    """Measure every requested role (default: all six, :data:`lobes.roles.ROLES`).

    Never raises — each role's measurement degrades independently (a dead
    ``senses`` backend doesn't stop ``cortex`` from being measured).
    """
    wanted = roles if roles is not None else ROLES
    return {role: measure_role(role, registry[role], timeout=timeout) for role in wanted}
