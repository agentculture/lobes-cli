"""The gateway HTTP server: a stdlib reverse proxy fronting the fleet backends.

``ThreadingHTTPServer`` + ``BaseHTTPRequestHandler``; the only module that opens
sockets. Routing *decisions* live in :func:`handle_post` (a seam that takes an
``open_upstream`` callable, so it's unit-testable without sockets) and in
:mod:`lobes.gateway._routing` (pure). The handler just reads the request,
calls :func:`handle_post`, and relays the chosen upstream response — buffered for
normal JSON, re-chunked for SSE streams.

**No cross-backend failover** (issue #91, "advertised implies reachable").
:func:`lobes.gateway._routing.order_backends` resolves a requested model to its
ONE owning backend; a model is never retried against a different backend serving
a different model (that would either 404 on an unknown id or, worse, silently
answer as the wrong model — a role-contract violation). Because the owner is the
only backend that can serve the model, its verdict is authoritative:

* a **2xx / 4xx** commits and is relayed verbatim — a 4xx (e.g. the owner's own
  404 "model does not exist") is a genuine *client* error now, not a trigger to
  fail over;
* a **refusal / timeout / >=500** means the owner is transiently down → a
  RETRYABLE **503** ``backend_unavailable`` + ``Retry-After`` (issue #14), NOT a
  terminal 404/502, so a client retries the same model instead of concluding it
  is gone;
* a **429** ``server_busy`` is the separate pressure-shed path (#85), and a
  **502** ``upstream_unavailable`` survives only for the degenerate malformed
  routing table (``order_backends`` returned an empty list) — see
  :func:`handle_post`.

Readiness governs *advertisement*, not routing: ``GET /v1/models`` and
``GET /capabilities`` fold in the background :class:`~lobes.gateway._readiness.
ReadinessCache` so a wired-but-dead backend is not advertised (issue #92); the
POST hot path never probes (it reads the socket-free cache, if at all).
"""

from __future__ import annotations

import dataclasses
import http.client
import json
import os
import sys
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Iterable
from urllib.parse import urlsplit

from lobes import _metrics
from lobes.catalog import as_dicts as supported_models_catalog
from lobes.gateway._config import ServerConfig
from lobes.gateway._pressure_policy import BUSY_RETRY_AFTER_SECONDS, decide
from lobes.gateway._readiness import ReadinessCache
from lobes.gateway._routing import (
    Backend,
    RoutingTable,
    is_audio_path,
    list_models_payload,
    order_backends,
    resolve_model,
    supported_models_payload,
)
from lobes.gateway._tier_request import (
    PressureCache,
    is_tier_alias,
    resolve_tier_request,
)

# NOTE: lobes.roles is imported lazily inside capabilities_payload() below, not
# here at module scope. lobes.roles itself imports lobes.gateway._config (for
# ServerConfig/build_config), and this package's own __init__.py imports THIS
# module (`from lobes.gateway.server import serve`) — a genuine import cycle.
# It only "worked" at module scope when something else happened to import
# lobes.gateway (fully) before anything imported lobes.roles first; entering
# via lobes.roles directly (e.g. `import lobes.roles_measure`) hit a partially
# initialized lobes.roles module and raised ImportError. Deferring the import
# to call time breaks the cycle without reordering either module.

_CHUNK = 65536

# Hop-by-hop headers must not be forwarded across a proxy (RFC 7230 §6.1). We also
# drop Content-Length/Transfer-Encoding in both directions and recompute framing.
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)


# --- request-body helpers (pure, testable) ---------------------------------


def _parse_body(body: bytes) -> dict | None:
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def extract_model(body: bytes) -> str | None:
    """The request's ``model`` field, or ``None`` (missing / malformed JSON)."""
    data = _parse_body(body)
    model = data.get("model") if data else None
    return model if isinstance(model, str) and model else None


def is_streaming(body: bytes) -> bool:
    """True when the request asked for an SSE stream (``"stream": true``)."""
    data = _parse_body(body)
    return bool(data and data.get("stream") is True)


def rewrite_model(body: bytes, served_name: str) -> bytes:
    """Rewrite the body's ``model`` to ``served_name`` so the backend accepts it.

    Aliases and default-routing change the model the *gateway* picked; the
    backend only knows its own ``--served-model-name``, so the forwarded body
    must carry that name. Non-JSON bodies pass through untouched.
    """
    data = _parse_body(body)
    if data is None:
        return body
    data["model"] = served_name
    return json.dumps(data).encode("utf-8")


def filter_headers(headers: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """Drop hop-by-hop headers (used both for the forwarded request and response)."""
    return [(k, v) for k, v in headers if k.lower() not in _HOP_BY_HOP]


# The request header that forces the requested tier despite pressure (t6, #68).
OVERRIDE_HEADER = "X-Lobes-Override"
_CONTENT_TYPE_JSON = "application/json"
_OVERRIDE_TRUTHY = frozenset({"1", "true", "yes"})


def is_override(value: str | None) -> bool:
    """True when ``X-Lobes-Override`` holds a truthy token (``1``/``true``/``yes``)."""
    return bool(value) and value.strip().lower() in _OVERRIDE_TRUTHY


def frame_chunk(chunk: bytes) -> bytes:
    """Wrap ``chunk`` in HTTP chunked-transfer framing (``<hex-len>\\r\\n<data>\\r\\n``)."""
    return b"%X\r\n" % len(chunk) + chunk + b"\r\n"


CHUNK_TERMINATOR = b"0\r\n\r\n"


def read_chunked_body(rfile, max_bytes: int = 64 * 1024 * 1024) -> bytes:
    """Decode an HTTP/1.1 ``Transfer-Encoding: chunked`` request body from ``rfile``.

    Clients/proxies may send a chunked body with no ``Content-Length``; reading
    only by length would forward an empty payload. Stops at the zero-length
    chunk, ignores chunk extensions, and caps the total at ``max_bytes`` so a
    malformed/huge stream can't exhaust memory.
    """
    body = bytearray()
    while len(body) <= max_bytes:
        size_line = rfile.readline()
        if not size_line:
            break  # stream ended early
        size_field = size_line.split(b";", 1)[0].strip()  # drop chunk extensions
        try:
            size = int(size_field, 16)
        except ValueError:
            break  # malformed size → stop rather than misread
        if size == 0:
            rfile.readline()  # consume the trailing CRLF after the last chunk
            break
        body += rfile.read(size)
        rfile.readline()  # consume the CRLF following each chunk
    return bytes(body)


# --- upstream client -------------------------------------------------------


class UpstreamError(Exception):
    """Connecting to a backend failed before any response (→ try the next one)."""


@dataclass
class _Upstream:
    """An opened upstream response. Duck-typed: tests substitute their own."""

    status: int
    headers: list[tuple[str, str]]
    _resp: object  # http.client.HTTPResponse
    _conn: object  # http.client.HTTPConnection

    def read(self, n: int) -> bytes:
        return self._resp.read(n)

    def read_all(self) -> bytes:
        return self._resp.read()

    def close(self) -> None:
        try:
            self._conn.close()
        except OSError:
            pass


def open_upstream(
    backend: Backend,
    path: str,
    body: bytes,
    headers: list[tuple[str, str]],
    *,
    connect_timeout: float,
    read_timeout: float,
) -> _Upstream:
    """POST ``body`` to ``backend`` and return the opened response.

    Uses a short ``connect_timeout`` for establishing the socket (so a down
    backend fails over fast) then a long ``read_timeout`` for the response (a
    reasoning model's first token is slow). Raises :class:`UpstreamError` if the
    backend can't be reached — including a malformed ``base_url`` (a non-numeric
    port makes ``parts.port`` raise ``ValueError``; a bad path/host raises
    ``http.client.InvalidURL``) — so the caller fails over instead of 500ing. An
    HTTP error *status* is returned as a normal response (the caller decides
    whether a 5xx triggers failover).
    """
    conn = None
    try:
        parts = urlsplit(backend.base_url)
        if parts.scheme == "https":
            conn = http.client.HTTPSConnection(
                parts.hostname, parts.port or 443, timeout=connect_timeout
            )
        else:
            conn = http.client.HTTPConnection(
                parts.hostname, parts.port or 80, timeout=connect_timeout
            )
        conn.connect()
        if conn.sock is not None:
            conn.sock.settimeout(read_timeout)
        conn.request("POST", path, body=body, headers=dict(headers))
        resp = conn.getresponse()
    except (OSError, http.client.HTTPException, ValueError) as exc:
        if conn is not None:
            conn.close()
        raise UpstreamError(f"{backend.name}: {exc}") from exc
    return _Upstream(
        status=resp.status, headers=filter_headers(resp.getheaders()), _resp=resp, _conn=conn
    )


OpenUpstream = Callable[..., _Upstream]


# --- routing + failover decision (pure seam) -------------------------------


@dataclass
class GatewayResponse:
    """What the handler should send. Either a gateway-generated body, or an
    upstream to relay (buffered or streaming)."""

    status: int
    headers: list[tuple[str, str]]
    body: bytes | None = None
    upstream: _Upstream | None = None
    streaming: bool = False
    attempts: list[str] = field(default_factory=list)


# Retry-After (seconds) on the 503 a transiently-down owner yields. The owner is
# the ONLY backend that can serve the requested model (#91: no failover), so its
# refusal / timeout / 5xx is a "come back shortly", not a terminal "no such model"
# — a caller should retry. Mirrors BUSY_RETRY_AFTER_SECONDS (the 429 shed) and the
# audio 503 (both 5s).
BACKEND_UNAVAILABLE_RETRY_AFTER_SECONDS: int = 5


def _error_body(
    message: str, attempts: list[str], *, error_type: str = "upstream_unavailable"
) -> bytes:
    """OpenAI-shaped gateway error body. ``error_type`` names the failure class a
    client must react to differently — the four are deliberately distinct:

    * ``upstream_unavailable`` — the degenerate **502**: ``order_backends``
      returned no owner (a malformed routing table). A config/deploy bug, not
      retryable.
    * ``backend_unavailable``  — the **503**: the one backend that owns this model
      refused / timed out / 5xx'd (#14/#91). Retryable (carries ``Retry-After``).
    * ``server_busy``          — the **429** pressure shed (#85), built separately
      by :func:`_busy_body`.
    * a relayed upstream ``404`` "model does not exist" — the owner's own verdict,
      never generated here.
    """
    return json.dumps(
        {"error": {"message": message, "type": error_type, "attempts": attempts}}
    ).encode("utf-8")


def _busy_body(requested_tier: str) -> bytes:
    """Return the JSON body for a 429 busy (shed) response."""
    _LANE_LABELS = {"main": "cortex", "multimodal": "senses"}
    label = _LANE_LABELS.get(requested_tier, requested_tier)
    return json.dumps(
        {
            "error": {
                "message": f"{label} is under pressure; retry shortly",
                "type": "server_busy",
                "code": "busy",
            }
        }
    ).encode("utf-8")


def handle_post(
    table: RoutingTable,
    cfg: ServerConfig,
    path: str,
    req_headers: Iterable[tuple[str, str]],
    body: bytes,
    open_upstream: OpenUpstream,
    *,
    pressure: dict[str, float] | None = None,
    override: bool = False,
) -> GatewayResponse:
    """Resolve the model to its ONE owning backend and try it exactly once.

    There is **no cross-backend failover** (issue #91):
    :func:`lobes.gateway._routing.order_backends` returns at most one backend —
    the owner of the resolved model — so this attempts that owner and nothing
    else. ``open_upstream`` is injected so this is unit-testable without sockets.

    The owner's verdict is authoritative, and the status mapping reflects that a
    request naming a model has exactly one honest place to go:

    * **2xx / 4xx** → commit to the owner and relay verbatim. A 4xx is a genuine
      *client* error (the owner is the only backend that could serve this model,
      so e.g. its 404 "model does not exist" is authoritative — never a reason to
      try someone else).
    * **refusal / timeout / >=500** (the loop exhausts its single attempt) → the
      owner is transiently down, so return a RETRYABLE **503** ``backend_unavailable``
      + ``Retry-After`` (issue #14). It is deliberately NOT a 404 (which would be
      indistinguishable from "this model id was never valid") and NOT a 502.
    * **empty ``order_backends``** (no owner at all) → the only remaining **502**
      ``upstream_unavailable``: a malformed routing table, a config bug, not
      retryable.

    Pressure-aware busy shedding (#85): when ``pressure`` is supplied *and* the
    requested model is a capability tier (``main``/``minor``/``multimodal``, or the
    ``cheap``/``normal``/``hard`` back-compat aliases), the tier is run through
    :func:`resolve_tier_request` *in front of* :func:`resolve_model`. Under
    memory/iowait pressure a ``main`` (cortex) or ``multimodal`` (senses) request
    is **shed** with HTTP 429 + ``Retry-After`` + ``X-Lobes-Tier-Reason: busy``
    and an OpenAI-shaped ``server_busy`` error body; no upstream is dialed. An
    explicit ``minor`` request is the floor and is still served (never shed). The
    ``X-Lobes-Override`` header (passed as ``override``) forces the requested tier
    to be served instead of shed. On the served path the ``X-Lobes-Tier`` /
    ``X-Lobes-Tier-Reason`` headers still travel with the response (prepended,
    streaming-safe). A plain model id, or ``pressure=None``, takes the existing
    non-tier path unchanged.
    """
    requested = extract_model(body)
    tier_headers: list[tuple[str, str]] = []
    if pressure is not None and is_tier_alias(requested):
        decision = resolve_tier_request(requested, pressure, override, table)
        if decision["busy"]:
            return GatewayResponse(
                status=429,
                headers=[
                    ("Retry-After", str(BUSY_RETRY_AFTER_SECONDS)),
                    ("X-Lobes-Tier-Reason", "busy"),
                    ("Content-Type", _CONTENT_TYPE_JSON),
                ],
                body=_busy_body(decision["requested_tier"]),
            )
        served = decision["served_name"]
        tier_headers = [
            ("X-Lobes-Tier", decision["served_tier"]),
            ("X-Lobes-Tier-Reason", decision["reason"]),
        ]
    else:
        served = resolve_model(table, requested)
    streaming = is_streaming(body)
    fwd_body = rewrite_model(body, served)
    fwd_headers = filter_headers(req_headers)
    attempts: list[str] = []

    ordered = order_backends(table, served)
    if not ordered:
        # DEGENERATE case ONLY: no backend owns `served` AND none owns
        # default_model — order_backends can return an empty list solely for a
        # malformed routing table (in practice, one with no primary). That is a
        # config/deploy bug, not a transient outage, so it is a TERMINAL 502
        # upstream_unavailable with NO Retry-After — never the retryable 503 a
        # present-but-dead owner gets below.
        return GatewayResponse(
            status=502,
            headers=tier_headers + [("Content-Type", _CONTENT_TYPE_JSON)],
            body=_error_body("no backend owns the requested model", attempts),
            attempts=attempts,
        )

    for backend in ordered:  # exactly one backend — no failover chain (#91)
        try:
            up = open_upstream(
                backend,
                path,
                fwd_body,
                fwd_headers,
                connect_timeout=cfg.connect_timeout,
                read_timeout=cfg.read_timeout,
            )
        except UpstreamError as exc:
            attempts.append(str(exc))
            continue
        if up.status >= 500:
            attempts.append(f"{backend.name}: HTTP {up.status}")
            up.close()
            continue
        # 2xx or 4xx → commit to the owner and relay verbatim. A 4xx is a genuine
        # CLIENT error: the owner is the only backend that could serve this model.
        return GatewayResponse(
            status=up.status,
            headers=tier_headers + up.headers,
            upstream=up,
            streaming=streaming,
            attempts=attempts,
        )

    # The single owner refused / timed out / 5xx'd. With no failover (#91) it is
    # the ONLY backend that could serve `served`, so this is a TRANSIENT owner-down
    # state — not "model unknown". Return a retryable 503 + Retry-After whose type
    # (backend_unavailable) is distinguishable from both the 429 server_busy shed
    # and the degenerate 502 upstream_unavailable above, so a client retries the
    # same model instead of treating the failure as terminal (issues #14, #91).
    return GatewayResponse(
        status=503,
        headers=tier_headers
        + [
            ("Retry-After", str(BACKEND_UNAVAILABLE_RETRY_AFTER_SECONDS)),
            ("Content-Type", _CONTENT_TYPE_JSON),
        ],
        body=_error_body(
            "the backend serving this model is unavailable — retry shortly",
            attempts,
            error_type="backend_unavailable",
        ),
        attempts=attempts,
    )


def handle_audio_post(
    cfg: ServerConfig,
    path: str,
    req_headers: Iterable[tuple[str, str]],
    body: bytes,
    open_upstream: OpenUpstream,
    *,
    audio_ready: bool | None = None,
) -> GatewayResponse:
    """Proxy an ``/v1/audio/*`` POST to the fixed audio backend.

    Unlike :func:`handle_post` this does **no** model parse/rewrite and **no**
    failover: the body is multipart (transcriptions) or TTS JSON (speech) and is
    forwarded verbatim to the one audio backend, whose response (a whole audio
    file or a small JSON) is relayed **streamed** (chunked). Returns 404 when no
    audio backend is configured (a text-only fleet leaves ``AUDIO_URL`` unset).
    ``open_upstream`` is injected so this is unit-testable without sockets.

    ``audio_ready`` is the caller's live readiness probe (issue #89): a value of
    ``False`` means the backend is reachable but still warming (Chatterbox/
    Parakeet loading, or a poisoned CUDA context) — we return a clear **503**
    with ``Retry-After`` instead of forwarding into a bare relayed 502, so a
    client can tell "not yet" from "broken". ``True``/``None`` forward as normal
    (``None`` = unreachable/unknown → the forward surfaces the honest 502).
    """
    if not cfg.audio_url:
        return GatewayResponse(
            status=404,
            headers=[("Content-Type", _CONTENT_TYPE_JSON)],
            body=_error_body("audio endpoints are not configured on this deployment", []),
        )
    if audio_ready is False:
        # Reachable but not ready — Chatterbox/Parakeet still warming up, or a
        # transient backend error its /v1/health/ready reported. A retryable 503,
        # distinct from the 502 an *unreachable* backend gets below.
        return GatewayResponse(
            status=503,
            headers=[("Content-Type", _CONTENT_TYPE_JSON), ("Retry-After", "5")],
            body=_error_body("audio backend not ready yet (warming up) — retry shortly", []),
        )
    backend = Backend(name="audio", base_url=cfg.audio_url, served_name="")
    fwd_headers = filter_headers(req_headers)
    try:
        up = open_upstream(
            backend,
            path,
            body,
            fwd_headers,
            connect_timeout=cfg.connect_timeout,
            read_timeout=cfg.read_timeout,
        )
    except UpstreamError as exc:
        return GatewayResponse(
            status=502,
            headers=[("Content-Type", _CONTENT_TYPE_JSON)],
            body=_error_body("audio backend is unavailable", [str(exc)]),
        )
    # 2xx, 4xx or 5xx — relay whatever the single audio backend says (no failover).
    # Stream the body through (chunked) rather than read_all()'ing it: a TTS WAV
    # can be many MB, and the gateway is the fleet's single front door — buffering
    # every audio response whole would let one large synthesis exhaust its memory.
    # up.headers is already hop-by-hop-filtered by open_upstream (Content-Length /
    # Transfer-Encoding dropped), so the chunked relay frames cleanly.
    return GatewayResponse(status=up.status, headers=up.headers, upstream=up, streaming=True)


# --- fleet status (the live aggregate the CLI can't get otherwise) ---------

# In the fleet the backends are internal-only (no host port), so only the gateway
# can see their /health + /metrics. This endpoint fans out and aggregates them into
# one JSON the host-side `lobes overview --live` renders. The prober is injected so
# this is unit-testable without sockets.

# Per-backend probe timeout for /status: bounded + probed in parallel (below) so a
# slow/down backend can't make the whole /status call hang for connect_timeout × N.
_STATUS_PROBE_TIMEOUT = 3.0


def _endpoints_for(table: RoutingTable, audio: bool) -> list[str]:
    """OpenAI endpoints this gateway actually serves, by the task families present."""
    tasks = {b.task for b in table.backends}
    eps = [
        "GET /health",
        "GET /status",
        "GET /v1/models",
        "GET /v1/models/supported",
        "GET /capabilities",
        "POST /v1/chat/completions",
        "POST /v1/completions",
    ]
    if "embed" in tasks:
        eps.append("POST /v1/embeddings")
    if "score" in tasks:
        eps += ["POST /v1/rerank", "POST /v1/score"]
    if audio:
        eps += ["POST /v1/audio/transcriptions", "POST /v1/audio/speech"]
    return eps


def fleet_status_payload(
    table: RoutingTable,
    cfg: ServerConfig,
    pressure: dict | None = None,
    probe=_metrics.probe_backend,
) -> dict:
    """Live status for every backend + an aggregate busy count + the endpoint list.

    Backends are probed **in parallel** with a bounded timeout, so a slow/down
    backend can't make ``/status`` hang for ``timeout × N``. ``base_url`` is
    intentionally **not** in the payload — those are internal-only routing details
    and ``/status`` may be reached over a public tunnel.

    When *pressure* is supplied (the cached ``/proc`` sample), a ``pressure``
    block is added exposing the busy-policy state a full-tier request would hit
    right now — ``mode`` (``warm``/``busy``), whether it is ``shed`` (HTTP 429),
    the ``reason`` and the raw swap/iowait numbers — so operators can see *why*
    callers are being told to wait (#85). Omitted entirely when *pressure* is
    ``None`` (no cache wired), keeping the payload back-compatible.
    """
    members = list(table.backends)
    if members:
        with ThreadPoolExecutor(max_workers=len(members)) as pool:
            results = list(
                pool.map(lambda b: probe(b.base_url, timeout=_STATUS_PROBE_TIMEOUT), members)
            )
    else:
        results = []
    backends: list[dict] = []
    running = waiting = 0
    for b, st in zip(members, results):
        metrics = st.get("metrics") or {}
        running += int(metrics.get("running", 0) or 0)
        waiting += int(metrics.get("waiting", 0) or 0)
        backends.append(
            {
                "name": b.name,
                "task": b.task,
                "served_name": b.served_name,
                "health": st.get("health", "unreachable"),
                "metrics": st.get("metrics"),
            }
        )
    payload = {
        "object": "lobes.fleet_status",
        "default_model": table.default_model,
        "busy": {"running": running, "waiting": waiting},
        "backends": backends,
        "endpoints": _endpoints_for(table, bool(cfg.audio_url)),
    }
    if pressure is not None:
        # Same decide() handle_post consults, probed with a full tier ("main"),
        # so the reported busy state matches what a live request would receive.
        d = decide(
            pressure.get("swap_used_percent", 0.0),
            pressure.get("iowait_percent", 0.0),
            requested_tier="main",
        )
        payload["pressure"] = {
            "mode": d["mode"],
            "shed": d["shed"],
            "reason": d["reason"],
            "swap_used_percent": pressure.get("swap_used_percent", 0.0),
            "iowait_percent": pressure.get("iowait_percent", 0.0),
        }
    return payload


# --- role capabilities (the #81 role→endpoint contract) --------------------

# GET /capabilities reuses lobes.roles.build_role_registry — the SAME builder
# the CLI's `lobes capabilities --json` calls — so the two payloads are
# exactly the same shape: a dict keyed by role, each value the full RoleInfo
# field set. The route derives a client-reachable origin (#87) and a live audio
# readiness signal (#89) and hands them to the builder; the pure function keeps
# its config-derived defaults so the CLI/unit path is unchanged.


def reachable_origin(
    host_header: str | None, public_url: str | None, scheme: str = "http"
) -> str | None:
    """The client-reachable gateway origin to advertise in /capabilities (#87).

    Prefers an explicit ``GATEWAY_PUBLIC_URL`` (``public_url``) — for a tunnel or
    a Host-rewriting reverse proxy — else echoes the origin the client actually
    dialed, taken from the request ``Host`` header (which already carries
    ``host:port`` in the right shape, IPv6 brackets included). Returns ``None``
    when neither is available, so the caller falls back to the config-derived
    origin (unchanged behaviour).
    """
    if public_url:
        return public_url.rstrip("/")
    if host_header:
        return f"{scheme}://{host_header}"
    return None


def _default_ready_probe(url: str, timeout: float) -> int:  # pragma: no cover - opens a socket
    parts = urlsplit(url)
    conn = http.client.HTTPConnection(parts.hostname, parts.port or 80, timeout=timeout)
    try:
        conn.request("GET", parts.path or "/")
        return conn.getresponse().status
    finally:
        conn.close()


def probe_audio_ready(
    audio_url: str,
    *,
    timeout: float = _STATUS_PROBE_TIMEOUT,
    opener: Callable[[str, float], int] | None = None,
) -> bool | None:
    """Live-probe the audio backend's aggregate readiness (issue #89).

    GETs ``<audio_url>/v1/health/ready`` (the realtime bridge's aggregate over
    Chatterbox + Parakeet) and maps the result to a tri-state so /capabilities
    and the audio proxy can tell a *warming* backend from an *unreachable* one:

    * ``True``  — HTTP 200: backends ready → a client request will round-trip.
    * ``False`` — reached the backend but it answered non-200 (e.g. 503 while a
      backend warms up, or a poisoned CUDA context) → advertised, not yet ready.
    * ``None``  — could not reach the backend at all (refused / timeout) →
      readiness unknown; the proxy forwards and lets a real request surface 502.

    ``opener`` is injected so this is unit-testable without sockets; the default
    opens a bounded ``http.client`` GET.
    """
    get_status = opener or _default_ready_probe
    try:
        return get_status(audio_url.rstrip("/") + "/v1/health/ready", timeout) == 200
    except (OSError, http.client.HTTPException, ValueError):
        # Mirror open_upstream's guard: a malformed AUDIO_URL (a non-numeric port
        # makes urlsplit(...).port raise ValueError) or a broken HTTP exchange
        # (HTTPException) must degrade to "readiness unknown" (None), never crash
        # the GET /capabilities or POST /v1/audio/* handler that called us.
        return None


def _ready_iff_true(snapshot: Mapping[str, bool | None]) -> dict[str, bool]:
    """Collapse the readiness tri-state to a strict "ready iff True" bool map.

    The background :class:`~lobes.gateway._readiness.ReadinessCache` reports a
    backend as ``None`` when it is dead / missing / unreachable (a probe that
    never got a 200 — see that module's docstring). But
    :func:`lobes.roles._gateway_role` reads a ``None`` signal as "no live signal
    available → fall back to the coarse ``loaded`` proxy", which for a WIRED
    backend is ``True``. Those two ``None``s mean OPPOSITE things: the cache's is
    "definitely not reachable", roles.py's is "I don't know, assume configured".
    Passing the cache's ``None`` straight through would therefore advertise a
    wired-but-dead backend as ``ready=True`` in ``/capabilities`` — the exact
    issue #92 defect (honesty h14). This bridges the vocabularies: only an
    affirmative ``True`` stays ``True`` (ready); both ``None`` (unreachable) and
    ``False`` (unhealthy) become a definite ``False``. It mirrors the ``is True``
    filter :func:`lobes.gateway._routing.list_models_payload` applies, so both
    advertisement surfaces agree — a dead backend is advertised nowhere.
    """
    return {name: signal is True for name, signal in snapshot.items()}


def capabilities_payload(
    table: RoutingTable,
    cfg: ServerConfig,
    env: Mapping[str, str] | None = None,
    *,
    gateway_url: str | None = None,
    audio_ready: bool | None = None,
    backend_ready: Mapping[str, bool | None] | None = None,
) -> dict:
    """The six first-class roles (issue #81), resolved via the shared registry.

    ``env`` defaults to ``os.environ``. The fleet compose passes the served
    ``PRIMARY_MAX_MODEL_LEN`` / ``MULTIMODAL_MAX_MODEL_LEN`` /
    ``EMBED_MAX_MODEL_LEN`` / ``RERANK_MAX_MODEL_LEN`` into the gateway
    container's environment (they are otherwise only given to the gear
    containers), so the served-context overlay resolves each role's SERVED
    ``--max-model-len`` here; it falls back to the catalog native when a var is
    unset.

    ``gateway_url`` is the client-reachable origin every role's ``endpoint`` is
    built from (issue #87) — the HTTP route derives it from the request Host
    header / ``GATEWAY_PUBLIC_URL`` via :func:`reachable_origin`. When ``None``
    the builder derives it from ``cfg.host``/``cfg.port`` (the CLI/unit path,
    unchanged). ``audio_ready`` is the live stt/tts readiness signal (issue #89)
    from :func:`probe_audio_ready`; when ``None`` the builder falls back to the
    configured ``bool(audio_url)`` (again the CLI/unit path). ``backend_ready`` is
    the live readiness snapshot for the four gateway-fronted roles (issue #92),
    keyed by internal ``Backend`` name — exactly what
    :meth:`lobes.gateway._readiness.ReadinessCache.current` returns, so the HTTP
    route passes it straight through; when ``None`` each role's ``ready`` falls
    back to ``loaded`` (the CLI/unit path). All three signal kwargs default to
    ``None`` so this pure function's shape is unchanged for its non-HTTP callers.
    """
    from lobes.roles import ROLES, build_role_registry  # deferred — see the module-level NOTE

    resolved_env = os.environ if env is None else env
    registry = build_role_registry(
        table,
        cfg,
        env=resolved_env,
        gateway_url=gateway_url,
        audio_ready=audio_ready,
        backend_ready=backend_ready,
    )
    return {role: dataclasses.asdict(registry[role]) for role in ROLES}


# --- the HTTP handler ------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    """Bound to a ``table`` + ``server_config`` by :func:`_make_handler`."""

    # Set per-server by _make_handler (frozen dataclasses → safe to share).
    table: RoutingTable
    server_config: ServerConfig
    # Non-blocking host-pressure provider (t6). None → the tier-downgrade layer
    # is skipped and tier aliases resolve via the static table (the t5 path).
    pressure_cache: PressureCache | None = None
    # Non-blocking background readiness provider (issue #92). None → /v1/models
    # lists every wired backend and /capabilities readiness falls back to the
    # coarse `loaded` proxy (the offline/unit path). Read only via .current()
    # (socket-free); the POST hot path never touches it.
    readiness_cache: ReadinessCache | None = None
    # HTTP/1.1 so we can stream with chunked transfer encoding.
    protocol_version = "HTTP/1.1"

    # --- GET: /health, /status, /v1/models, /v1/models/supported ---
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        route = self.path.split("?", 1)[0]
        if route == "/health":
            self._send_json(200, {"status": "ok", "service": "model-gear-gateway"})
        elif route == "/status":
            # Live aggregate the host CLI can't get otherwise: the backends are
            # internal-only, so the gateway fans out to each one's /health + /metrics.
            # The cached pressure sample surfaces the busy-policy state (#85).
            pressure = self.pressure_cache.current() if self.pressure_cache is not None else None
            self._send_json(200, fleet_status_payload(self.table, self.server_config, pressure))
        elif route == "/v1/models":
            # Advertise only backends the live readiness snapshot marks ready
            # (issue #92): a wired-but-dead backend must NOT appear here, so a
            # client can trust that a listed model id reaches a live engine. The
            # snapshot is socket-free (.current() never probes); with no cache
            # wired, every backend is listed (the offline/unit path).
            ready = self.readiness_cache.current() if self.readiness_cache is not None else None
            self._send_json(200, list_models_payload(self.table, ready))
        elif route == "/v1/models/supported":
            # The full catalog of gears you can change to (loaded + the rest),
            # not just the two currently warm. Non-OpenAI shape; /v1/models stays standard.
            self._send_json(200, supported_models_payload(self.table, supported_models_catalog()))
        elif route == "/capabilities":
            # The #81 role→endpoint contract: SIX first-class roles resolved to
            # live metadata via the shared lobes.roles registry. The endpoint is
            # the client-reachable origin this request actually dialed (#87),
            # stt/tts readiness is a live probe of the audio backend (#89), and the
            # four gateway-fronted roles' readiness comes from the background
            # ReadinessCache snapshot (#92) — read socket-free, no probe here.
            cfg = self.server_config
            origin = reachable_origin(self.headers.get("Host"), cfg.public_url)
            audio_ready = probe_audio_ready(cfg.audio_url) is True if cfg.audio_url else None
            # Collapse the cache's tri-state to a strict "ready iff True" map before
            # handing it to the registry: at the gateway a None from the cache means
            # "probed, unreachable", NOT roles.py's "no signal → fall back to
            # loaded". Without this a wired-but-dead backend would advertise
            # ready=True here (issue #92 / honesty h14). See _ready_iff_true.
            backend_ready = (
                _ready_iff_true(self.readiness_cache.current())
                if self.readiness_cache is not None
                else None
            )
            self._send_json(
                200,
                capabilities_payload(
                    self.table,
                    cfg,
                    gateway_url=origin,
                    audio_ready=audio_ready,
                    backend_ready=backend_ready,
                ),
            )
        else:
            self._send_json(404, {"error": {"message": f"not found: {route}", "type": "not_found"}})

    # --- POST: proxy /v1/* to a backend ---
    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        body = self._read_body()
        if is_audio_path(self.path):
            # /v1/audio/* → the audio backend, path-routed (no model rewrite/failover).
            # Probe readiness first so a warming backend gets a clear 503, not a
            # bare relayed 502 (#89). audio_url unset → 404 inside handle_audio_post.
            cfg = self.server_config
            audio_ready = probe_audio_ready(cfg.audio_url) if cfg.audio_url else None
            resp = handle_audio_post(
                cfg,
                self.path,
                list(self.headers.items()),
                body,
                open_upstream,
                audio_ready=audio_ready,
            )
        else:
            # Read pressure from the cache (O(1), never samples here) and the
            # override header so the tier-downgrade layer runs in front of routing.
            pressure = self.pressure_cache.current() if self.pressure_cache is not None else None
            override = is_override(self.headers.get(OVERRIDE_HEADER))
            resp = handle_post(
                self.table,
                self.server_config,
                self.path,
                list(self.headers.items()),
                body,
                open_upstream,
                pressure=pressure,
                override=override,
            )
        if resp.upstream is None:
            self._send_simple(resp.status, resp.headers, resp.body or b"")
            return
        try:
            if resp.streaming:
                self._relay_streaming(resp)
            else:
                self._relay_buffered(resp)
        finally:
            resp.upstream.close()

    # --- relay helpers ---
    def _read_body(self) -> bytes:
        cl = self.headers.get("Content-Length")
        if cl is not None:
            try:
                length = int(cl)
            except ValueError:
                length = 0
            return self.rfile.read(length) if length > 0 else b""
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            return read_chunked_body(self.rfile)
        return b""

    def _relay_buffered(self, resp: GatewayResponse) -> None:
        data = resp.upstream.read_all()
        self.send_response(resp.status)
        for key, value in resp.headers:
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if data:
            self.wfile.write(data)

    def _relay_streaming(self, resp: GatewayResponse) -> None:
        self.send_response(resp.status)
        for key, value in resp.headers:
            self.send_header(key, value)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        while True:
            chunk = resp.upstream.read(_CHUNK)
            if not chunk:
                break
            self.wfile.write(frame_chunk(chunk))
            self.wfile.flush()  # SSE must flush per chunk or it buffers until EOF
        self.wfile.write(CHUNK_TERMINATOR)
        self.wfile.flush()

    def _send_json(self, status: int, obj: dict) -> None:
        self._send_simple(status, [("Content-Type", _CONTENT_TYPE_JSON)], json.dumps(obj).encode())

    def _send_simple(self, status: int, headers: list[tuple[str, str]], body: bytes) -> None:
        self.send_response(status)
        for key, value in headers:
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:  # keep request logs tidy in docker logs
        sys.stderr.write("[gateway] %s\n" % (fmt % args))


def _make_handler(
    table: RoutingTable,
    cfg: ServerConfig,
    pressure_cache: PressureCache | None = None,
    readiness_cache: ReadinessCache | None = None,
) -> type[_Handler]:
    bound = type(
        "_BoundHandler",
        (_Handler,),
        {
            "table": table,
            "server_config": cfg,
            "pressure_cache": pressure_cache,
            "readiness_cache": readiness_cache,
        },
    )
    return bound


def serve(table: RoutingTable, cfg: ServerConfig) -> None:  # pragma: no cover
    """Bind and serve forever (the long-lived gateway process)."""
    # One pressure cache per process: a background daemon thread refreshes it so
    # the 150 ms sample never lands on the request path.
    pressure_cache = PressureCache()
    # One readiness cache per process (issue #92). Construction seeds every backend
    # to None (unknown) WITHOUT probing, so we do ONE bounded synchronous refresh
    # BEFORE binding — otherwise /v1/models would advertise nothing until the
    # daemon's first background pass lands (up to one interval), reporting a false
    # "fleet is empty" on the very first request. After the seed, start() hands
    # refreshes to a background daemon thread so no probe ever lands on the request
    # path. Read verbs consult it via .current() (socket-free); the POST hot path
    # never touches it.
    readiness_cache = ReadinessCache.from_backends(table.backends, start=False)
    readiness_cache.refresh()
    readiness_cache.start()
    httpd = ThreadingHTTPServer(
        (cfg.host, cfg.port), _make_handler(table, cfg, pressure_cache, readiness_cache)
    )
    sys.stderr.write(f"[gateway] listening on {cfg.host}:{cfg.port}\n")
    httpd.serve_forever()
