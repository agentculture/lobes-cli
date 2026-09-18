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

**Inbound auth** (proxy-lobes t2, issues #115/#127): with
:attr:`~lobes.gateway._config.ServerConfig.api_key` set (``GATEWAY_API_KEY`` →
``CULTURE_VLLM_API_KEY``, resolved in t1) the handler gates every DATA-PLANE
route on ``Authorization: Bearer <key>`` — see the "inbound bearer auth"
section below for the route policy, the timing-safe comparison, and the
never-echo-key-material contract. With ``api_key`` unset (the default) the
gate is provably inert: no header is ever inspected and every route behaves
byte-identically to the pre-auth gateway. The gate is the INBOUND edge only —
outbound header forwarding to local backends is unchanged.

**The proxy data plane** (proxy-lobes t6, issues #115/#127) is the third lobe
state — awake (hosted) / asleep (referral-only 404) / **PROXY**: a dropped
role opted in to proxying (``table.peer_proxied``) is answered by FORWARDING
the request to the declared peer origin, replacing the referral 404 for
exactly those names and nothing else. See the "proxy data plane" section
below for the loop guard, the pairwise-credential swap, the failure modes,
and why proxied requests bypass the LOCAL pressure policy. Retired (t14): the
env peer family that used to populate ``table.peer_proxied`` is gone — the
mesh RoutingSnapshot (t13) is the candidate source now — so this field is
always empty in practice; the mechanics below are unchanged and dormant.
"""

from __future__ import annotations

import hmac
import http.client
import json
import os
import re
import socket
import sys
import threading
import uuid
from collections import OrderedDict
from collections.abc import Collection, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Iterable
from urllib.parse import quote, urlencode, urlsplit

from lobes import __version__, _metrics
from lobes.catalog import SUPPORTED_MODELS
from lobes.catalog import as_dicts as supported_models_catalog
from lobes.gateway._authlog import RejectionLog, rejection_reason
from lobes.gateway._config import (
    NEVER_PROXIED_BACKENDS,
    RENDER_BODY_LIMIT_ENV,
    ServerConfig,
)
from lobes.gateway._mesh_config import MeshConfigError
from lobes.gateway._mesh_config import build_mesh_config as _build_mesh_config
from lobes.gateway._mesh_routes import (
    MeshRoutes,
    announcement_from_capabilities,
)
from lobes.gateway._mesh_routes import build_mesh_routes as _build_mesh_routes
from lobes.gateway._mesh_routes import (
    dispatch_mesh,
)
from lobes.gateway._mesh_routes import is_mesh_route as _is_mesh_route
from lobes.gateway._mesh_routes import require_self_origin as _require_self_origin
from lobes.gateway._mesh_routes import start_mesh as _start_mesh
from lobes.gateway._mesh_routing import (
    MESH_MEMBER_HEADER,
    MeshRoutingView,
    RolePlacement,
    RoutingSnapshot,
    SnapshotHolder,
    as_routing_snapshot,
    build_snapshot,
    compute_role_placement,
    find_suffixed_lane,
    mesh_markers,
)
from lobes.gateway._mesh_wire import Announcement
from lobes.gateway._pressure_policy import BUSY_RETRY_AFTER_SECONDS, decide
from lobes.gateway._readiness import PeerSpec, ReadinessCache
from lobes.gateway._realtime import (
    HandshakeError,
    RealtimeRefusal,
    is_realtime_path,
    plan_realtime_upgrade,
    read_head,
    run_tunnel,
    status_of,
    upgrade_request_bytes,
)
from lobes.gateway._replicas import (
    PEER_CAPACITY_KEY,
    LocalLane,
    PeerReplica,
    ReplicaCache,
    ReplicaState,
)
from lobes.gateway._routing import (
    RENDER_PATH,
    RENDER_TASK,
    Backend,
    RenderRoute,
    RoutingTable,
    audio_role_for_path,
    extract_job_artifacts,
    infeasible_owner,
    is_audio_path,
    is_render_path,
    is_unknown_model,
    list_models_payload,
    order_backends,
    parse_render_route,
    resolve_model,
    supported_models_payload,
)
from lobes.gateway._selection import (
    REASON_NONE,
    REASON_SOLE_READY,
    Selection,
    is_calibrated,
    select_replica,
    selection_capacity,
    selection_wait,
)
from lobes.gateway._tier_request import (
    PressureCache,
    is_tier_alias,
    resolve_tier_request,
)
from lobes.roles import BACKEND_ROLE, role_registry_from_env

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


# --- mesh snapshot helpers --------------------------------------------------


def _first_stt_origin(snapshot: "RoutingSnapshot | None") -> str | None:
    """Return the first verified stt origin, or first announced-only stt origin."""
    if snapshot is None:
        return None
    origins = snapshot.member_origins("stt")
    if origins:
        return origins[0]
    for m in snapshot.members:
        if "stt" in m.announced_roles:
            return m.origin
    return None


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


# --- inbound bearer auth (opt-in via GATEWAY_API_KEY, issues #115/#127) -----

# Until proxy-lobes t2 the gateway forwarded the caller's Authorization header
# to upstreams but never VALIDATED it inbound (the known limitation documented
# in docs/gateway-fleet.md). With ServerConfig.api_key set (t1's config
# channel: GATEWAY_API_KEY → CULTURE_VLLM_API_KEY → None) the handler now
# gates the DATA PLANE at its inbound edge:
#
# * every **POST** route — chat/completions, completions, embeddings, rerank,
#   score, audio/* … every POST the gateway answers is a forward to a backend,
#   so the whole method is data plane;
# * the **GET /v1/*** namespace — /v1/models and /v1/models/supported are part
#   of the OpenAI surface callers script against (they enumerate this
#   deployment's served models), and gating the whole /v1/ GET namespace also
#   means an unauthenticated caller learns nothing about which /v1 routes
#   exist (401 outranks the 404).
#
# Other HTTP methods (HEAD/OPTIONS/PUT/…) need no gate: the handler implements
# only do_GET/do_POST, so BaseHTTPRequestHandler answers every other method
# with its stock 501 before ANY routing, body read, or backend logic runs —
# the same pre-auth contract the gateway always had, with nothing to leak.
#
# Two surfaces stay KEYLESS — a POLICY DECISION, not an omission:
#
# * ``/health`` is the container-probe endpoint: the compose healthcheck and
#   peer boxes must reach it before any key has been distributed, and a gated
#   healthcheck would mark the container unhealthy the moment a key is
#   configured — an auth knob must never masquerade as an outage.
# * ``/capabilities`` is the control-plane discovery/honesty surface (issues
#   #81/#112): peers and referral-followers read it to learn WHICH roles this
#   box hosts (and, via ``hosted_by``, where a dropped role lives) BEFORE they
#   hold any key; gating it would break the honest-referral contract for
#   exactly the callers it exists to serve.
#
# ``/status`` (the operator observability aggregate ``lobes overview --live``
# reads) is control-plane with them and stays keyless: it serves no inference
# and echoes no request/response data.
#
# The gate runs BEFORE the request body is read/parsed, before model
# resolution, before any readiness probe, and before any upstream connection —
# a rejected request costs the fleet zero sockets. With ``api_key`` unset the
# gate is provably inert (see _Handler._authorized: it returns before the
# Authorization header is even READ), so an untouched deployment is
# byte-identical to the pre-auth gateway on every route.

_WWW_AUTHENTICATE_HEADER = ("WWW-Authenticate", "Bearer")


def bearer_token_matches(api_key: str, authorization: str | None) -> bool:
    """True iff ``authorization`` is a well-formed ``Bearer`` credential whose
    token equals ``api_key``.

    Parsing is strict and fails CLOSED: the scheme must be ``Bearer``
    (case-insensitive, RFC 7235 §2.1) and the remainder — stripped of
    surrounding whitespace — must be non-empty. A missing header, a foreign
    scheme (``Basic …``), a bare token with no scheme, and an empty token are
    all rejected before any comparison happens.

    The token comparison is :func:`hmac.compare_digest` over **utf-8 bytes**:
    constant-time, so a caller probing the gateway cannot use response-timing
    differences to recover the key byte-by-byte (the standard remediation for
    a string-equality timing oracle; bytes rather than str because
    ``compare_digest`` is only timing-safe for ASCII-compatible str). Neither
    input is ever logged or echoed by any caller of this function — see
    :func:`_invalid_api_key_body`.
    """
    if not authorization:
        return False
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return False
    token = token.strip()
    if not token:
        return False
    return hmac.compare_digest(token.encode("utf-8"), api_key.encode("utf-8"))


def _invalid_api_key_body() -> bytes:
    """OpenAI-shaped 401 body for a missing / malformed / wrong inbound key.

    Mirrors the ``invalid_api_key`` error the OpenAI API emits, so a caller's
    existing error handling (e.g. openai-python raising ``AuthenticationError``
    on 401) works unchanged against the gateway. DELIBERATELY STATIC: the
    message names the fix (send ``Authorization: Bearer <key>``) but never
    distinguishes missing from malformed from wrong-key, and never echoes what
    the caller sent nor any part of the expected key — a 401 must not become a
    key-material oracle.
    """
    return json.dumps(
        {
            "error": {
                "message": (
                    "Invalid API key. Pass this gateway's configured key as "
                    "'Authorization: Bearer <key>'."
                ),
                "type": "invalid_api_key",
                "code": "invalid_api_key",
            }
        }
    ).encode("utf-8")


# --- force-strict-tools (GATEWAY_FORCE_STRICT_TOOLS, opt-in, colleague#320) -

# The cortex thinking model occasionally drifts off its tool-call template;
# vLLM's parser salvage then mangles the call (e.g. name='read_file"' + empty
# args). xgrammar structural-tag constrained decoding (OpenAI's `strict:
# true` on a tool's `function`) makes a malformed call impossible — this knob
# is how EXISTING callers get that without a client-side change. Default off
# (ServerConfig.force_strict_tools, from GATEWAY_FORCE_STRICT_TOOLS) is a
# hard byte-identical-passthrough guarantee: every helper below is only
# reachable from handle_post when the knob is truthy.

_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

# The generate lanes force-strict-tools may arm, by internal Backend name.
#
# `primary` (cortex) ONLY, deliberately — this knob exists for a defect measured
# on exactly that lane: the qwen3_coder structural-tag call site hardcoding
# `reasoning=False` (colleague#320), fixed by the qwen3_coder_thinking plugin.
#
# WHY MUSE IS NOT HERE, despite serving tool calls and declaring `tool_use`:
# because on that lane the knob is INERT. Measured live on the 31B (Thor,
# 2026-07-17, `--tool-call-parser=gemma4` + MTP), `strict: true` never engages
# xgrammar at all:
#   * a tool schema carrying a regex xgrammar cannot compile (a lookahead) is
#     accepted with HTTP 200 instead of raising a grammar-compile failure;
#   * the server logs no structural_tag/xgrammar/grammar line for such a request;
#   * output is byte-comparable with `strict: false`.
# Injecting `strict` here would therefore add a claim ("this lane is
# grammar-constrained") that the lane does not honour — the exact
# advertise-what-you-cannot-serve failure #92 exists to prevent.
#
# Two rationales an earlier draft of this comment gave, BOTH disproven live on
# 2026-07-17 — do not reinstate them:
#   * "Gemma4EngineToolParser declares supports_required_and_named = False" —
#     so does Qwen3EngineToolParser, the primary lane's own parser, which IS
#     armed here. The flag does not distinguish the two lanes.
#   * "forcing structured output crashes EngineCore under speculative decoding"
#     (from Gemma4EngineToolParser.adjust_request's docstring) — real for the
#     structured-outputs path that parser deliberately skips, but NOT reachable
#     via this knob: strict requests were served repeatedly with the engine
#     healthy afterwards. The crash risk was hypothetical, not measured.
#
# Widen this only with a live transcript showing strict decoding actually
# CONSTRAINS decoding on the target lane (#108) — a no-op is not a benefit.
_STRICT_TOOL_LANES: frozenset[str] = frozenset({"primary"})


def _is_chat_completions_request(path: str) -> bool:
    """True for the one endpoint force-strict-tools may touch — the chat
    lane. ``/v1/completions`` (legacy, no ``tools``), embeddings, rerank, and
    audio are never in scope."""
    return path.split("?", 1)[0] == _CHAT_COMPLETIONS_PATH


def _tools_present(data: dict) -> bool:
    tools = data.get("tools")
    return isinstance(tools, list) and len(tools) > 0


def inject_strict_tools(body: bytes) -> tuple[bytes, list[str]] | None:
    """Inject ``"strict": true`` into every ``tools[i].function`` that lacks
    an explicit ``strict`` key.

    Caller wins: a tool that already carries ANY ``strict`` value (``true``
    OR ``false``) is left untouched — only an ABSENT key is filled in. Pure
    and testable without sockets, matching the sibling body helpers above
    (:func:`rewrite_model` etc).

    Returns ``None`` — no injection performed — when the body is not JSON,
    carries no non-empty ``tools`` array, or every tool already declares its
    own ``strict`` (nothing was actually modified). :func:`handle_post` reads
    ``None`` as "this request is not eligible for the retry-without-strict
    fallback": a caller who set ``strict`` themselves and then hits a
    compile failure gets that failure as their own outcome, not a retry.
    """
    data = _parse_body(body)
    if data is None or not _tools_present(data):
        return None
    tool_names: list[str] = []
    for tool in data["tools"]:
        if not isinstance(tool, dict):
            continue
        func = tool.get("function")
        if not isinstance(func, dict) or "strict" in func:
            continue  # absent-only: an explicit strict (true OR false) wins
        func["strict"] = True
        name = func.get("name")
        tool_names.append(name if isinstance(name, str) and name else "<unnamed>")
    if not tool_names:
        return None  # nothing was actually modified — not retry-eligible
    return json.dumps(data).encode("utf-8"), tool_names


# Heuristic signature list for a strict-injection schema/grammar-compile
# failure — a HEURISTIC pending live discovery of vLLM's actual error text
# (devague plan risk r1). Matched case-insensitively as a bare substring
# against the upstream error body. Module-level so it is one place to widen
# once a real failure is observed on the live rig.
_STRICT_FAILURE_SIGNATURES: tuple[str, ...] = (
    "structural_tag",
    "xgrammar",
    "grammar",
    "json_schema",
)

_STRICT_RETRY_LOG_SNIPPET_LEN = 200


def _matches_strict_failure_signature(body: bytes) -> bool:
    text = body.decode("utf-8", errors="replace").lower()
    return any(sig in text for sig in _STRICT_FAILURE_SIGNATURES)


def _log_strict_retry(tool_names: list[str], upstream_body: bytes) -> None:
    """One log line naming the failing tool schema(s) + an upstream error
    snippet, via the module's existing stderr-logging pattern (see
    :meth:`_Handler.log_message` / :func:`serve`)."""
    snippet = upstream_body.decode("utf-8", errors="replace")[:_STRICT_RETRY_LOG_SNIPPET_LEN]
    names = ", ".join(tool_names) or "<none>"
    sys.stderr.write(
        f"[gateway] strict-tools compile failure for tool(s) [{names}] — "
        f"retrying without strict; upstream said: {snippet!r}\n"
    )


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

# --- terminal SSE frames for a stream that dies mid-flight (issue #220) -----
#
# An SSE stream has exactly one honest ending: the sentinel `data: [DONE]`
# event, then the zero-length chunk that closes HTTP chunked framing. Before
# #222/#220 the relay loop below sent NEITHER when it failed: an exception
# unwound out of `_relay_streaming` and the client — parked in a blocking read
# on a still-ESTABLISHED socket — had nothing to distinguish "the model is
# still thinking" from "the upstream died 20 minutes ago". Observed on the
# DGX Spark 2026-08-27/28: 5 of 15 streamed runs sat 17-24 minutes with
# `vllm:num_requests_running 0` and the GPU at 3-7% before being cut by hand.
#
# So: whatever happens, this handler ends the stream. A client that already
# hung up gets nothing (there is nowhere to write); an upstream that failed
# mid-stream gets an error event AND the sentinel, so a client that only knows
# how to look for `[DONE]` still terminates.
SSE_DONE = b"data: [DONE]\n\n"


def sse_error_frame(message: str) -> bytes:
    """One OpenAI-shaped SSE error event (``data: {"error": {...}}``).

    Shaped like the error body a non-streamed request would have received, so a
    caller can parse a mid-stream failure with the code it already has. The
    ``type``/``code`` are the gateway's own ``upstream_error`` — this is never
    the upstream's own error payload (by the time it fires the upstream has
    stopped producing bytes), and mislabelling it as one would be a lie.
    """
    payload = json.dumps(
        {
            "error": {
                "message": message,
                "type": "upstream_error",
                "code": "upstream_error",
            }
        }
    )
    return b"data: " + payload.encode("utf-8") + b"\n\n"


class RequestBodyTooLarge(Exception):
    """A request body exceeded the limit its route declares (→ HTTP 413).

    Carries what the refusal needs to be actionable: the size the caller
    DECLARED (``None`` on a chunked body, where nothing declares it up front),
    the limit it broke, and the name of the env knob that sets that limit — so
    the operator is told which value to raise rather than being left to guess
    between two per-route caps.
    """

    def __init__(self, declared: int | None, limit: int, knob: str) -> None:
        self.declared = declared
        self.limit = limit
        self.knob = knob
        super().__init__(f"request body over the {limit}-byte {knob} limit")


def read_chunked_body(rfile, max_bytes: int = 64 * 1024 * 1024, *, strict: bool = False) -> bytes:
    """Decode an HTTP/1.1 ``Transfer-Encoding: chunked`` request body from ``rfile``.

    Clients/proxies may send a chunked body with no ``Content-Length``; reading
    only by length would forward an empty payload. Stops at the zero-length
    chunk, ignores chunk extensions, and caps the total at ``max_bytes`` so a
    malformed/huge stream can't exhaust memory.

    ``strict`` is the caller's choice of what "over the cap" MEANS, and its
    default keeps every pre-existing call site byte-identical:

    * ``False`` (the default, and what every non-render lane still passes) —
      the total is TRUNCATED at ``max_bytes`` and the request proceeds, exactly
      as it did before the render lane grew limits.
    * ``True`` — :class:`RequestBodyTooLarge` is raised INCREMENTALLY, as soon
      as a chunk header declares bytes that would take the total past the cap,
      so the oversized payload is never read into memory at all. The caller is
      then responsible for closing the connection: the rest of the body is
      still on the socket and would poison the next request's framing.

    ``knob`` is not a parameter here — the strict caller
    (:meth:`_Handler._read_body`) re-raises with the route's own knob name.
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
        if strict and len(body) + size > max_bytes:
            # Refuse on the chunk HEADER, before those bytes are read: the whole
            # point is that an oversized body never lands in gateway memory.
            raise RequestBodyTooLarge(None, max_bytes, "")
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
        # read1, not read: read(n) blocks until n bytes or EOF, so a whole
        # SSE turn (a few KB) only returns at EOF. read1 returns as soon as
        # any bytes are available (b"" only at EOF), letting the relay loop
        # forward frames as they arrive instead of in one terminal burst.
        return self._resp.read1(n)

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
    method: str = "POST",
) -> _Upstream:
    """Send ``body`` to ``backend`` with ``method`` and return the opened response.

    ``method`` defaults to ``"POST"``, which is what every caller before the
    innereye work passed implicitly — a default, not a new decision at those
    call sites, so their behaviour is byte-identical. It is keyword-only, so no
    positional call site can be re-read by accident. A ``"GET"`` call takes the
    same socket treatment (below) and the same failure contract; the body is
    simply empty.

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
        conn.request(method, path, body=body, headers=dict(headers))
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
    # True ONLY on the retryable 503 a peer that refused / timed out / 5xx'd
    # BEFORE returning any bytes produces (:func:`_peer_unavailable_response`).
    # The replica pool's pre-dispatch retry (t8, #199, spec c15/h12) keys on
    # this rather than sniffing status codes, so a peer's own *relayed* 503 —
    # if one ever became relayable — could never be mistaken for "this replica
    # never answered" and re-issued against another box.
    peer_unavailable: bool = False
    # Work that must run when this answer has been fully DELIVERED, not when
    # it was built (capacity-relative pool routing, t5). Exactly one thing
    # uses it today: releasing the replica-pool in-flight counter for an
    # answer whose completion escapes the dispatch block — a relayed
    # ``upstream`` is a one-shot byte tunnel drained by the handler long after
    # `_pool_attempt` returned. A dispatch whose outcome DOES fit in that
    # block (a pre-dispatch failure, a gateway-generated body) is released
    # there and never reaches this field. ``None`` on every non-pooled
    # response, so :meth:`release` is a no-op for the whole pre-pool surface.
    on_complete: "Callable[[], None] | None" = None

    def release(self) -> None:
        """Run and forget :attr:`on_complete`. Idempotent by construction.

        Called from the handler's ``finally`` after the relay, so a caller
        that disconnects mid-stream releases the counter exactly as a clean
        completion does. Clearing the attribute FIRST means a double call (a
        handler ``finally`` plus a defensive call elsewhere) cannot
        double-release; the underlying ``end_dispatch`` is idempotent too, so
        this is the second of three independent guards against a leaked
        counter.
        """
        hook, self.on_complete = self.on_complete, None
        if hook is not None:
            hook()


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
    client must react to differently — the five are deliberately distinct:

    * ``upstream_unavailable`` — the degenerate **502**: ``order_backends``
      returned no owner (a malformed routing table). A config/deploy bug, not
      retryable.
    * ``backend_unavailable``  — the **503**: the one backend that owns this model
      refused / timed out / 5xx'd (#14/#91). Retryable (carries ``Retry-After``).
    * ``server_busy``          — the **429** pressure shed (#85), built separately
      by :func:`_busy_body`.
    * ``role_unverified``      — the boot-window **503**, built separately by
      :func:`_role_unverified_body`: the role's only mesh candidate announced
      it but has not been probed yet, so the honest answer is "not yet"
      (retryable, carries ``Retry-After``) rather than the terminal 404
      ``role_infeasible`` "never".
    * a relayed upstream ``404`` "model does not exist" — the owner's own verdict,
      never generated here.
    """
    return json.dumps(
        {"error": {"message": message, "type": error_type, "attempts": attempts}}
    ).encode("utf-8")


def _model_not_found_body(model: str) -> bytes:
    """OpenAI/vLLM-shaped 404 body for an id that was NEVER advertised (honesty h23).

    Mirrors the ``model_not_found`` error an OpenAI/vLLM backend emits for an
    unknown model, so a client sees a consistent 404 shape whether it hit the
    gateway or a backend directly. This is NOT a contradiction of "advertised
    implies reachable" (issue #92): the invariant is that a model *listed in
    ``/v1/models``* never 404s — an id that was never listed *should* 404. It is
    the deliberate converse of the never-404 race guarantee (see
    :func:`handle_post`).
    """
    return json.dumps(
        {
            "error": {
                "message": f"The model `{model}` does not exist.",
                "type": "model_not_found",
                "code": "model_not_found",
            }
        }
    ).encode("utf-8")


def _role_infeasible_body(
    requested: str | None,
    backend_name: str,
    peer_origin: str | None = None,
    *,
    suffixed_names: "tuple[str, ...] | None" = None,
) -> bytes:
    """4xx body for a request pinned to a HARDWARE-infeasible backend (t6).

    Distinct ``type``/``code`` from :func:`_model_not_found_body`: the
    requested id/role IS part of the role contract (it may even be
    wired — the primary is unconditionally wired regardless of feasibility)
    but this machine's per-machine profile declared its owning backend
    (``backend_name``) unable to serve it at all. Never a reason to
    silently substitute a different, feasible gear — see
    :func:`lobes.gateway._routing.infeasible_owner`.

    ``peer_origin`` is the opt-in honest referral (mesh-brain t3, issue
    #112): the origin of the peer box that hosts this role, now sourced from
    a verified mesh member rather than the retired env peer family (t14; the
    per-backend ``<PREFIX>_PEER_ORIGIN`` var that used to populate this no
    longer exists — see ``lobes.gateway._config``'s "Retired" comment). When set, the message
    names it and a machine-readable ``hosted_by`` key is added — a referral
    for the CALLER to dial directly; a REFERRAL-ONLY gateway never forwards
    the request there (data-plane forwarding exists only for names the
    operator additionally armed via ``table.peer_proxied``, which never reach
    this body — see :func:`_proxy_to_peer`). When ``None`` (no peer config —
    the default) the body is BYTE-IDENTICAL to the pre-referral contract.
    """
    label = requested or "(unspecified)"
    message = (
        f"The model `{label}` is not feasible on this machine — its "
        f"backend (`{backend_name}`) is declared hardware-infeasible "
        "by this deployment's per-machine profile and will never be "
        "served here."
    )
    error: dict[str, str] = {}
    if peer_origin:
        message += (
            f" It is hosted by the peer at `{peer_origin}` — address that box "
            "directly; this gateway never proxies requests to peers."
        )
    error["message"] = message
    error["type"] = "role_infeasible"
    error["code"] = "role_infeasible"
    if peer_origin:
        error["hosted_by"] = peer_origin
    # Suffixed-lane naming (t8, issue #237): a raw id/role hosted ONLY as
    # disagreeing mesh members is never silently resolved to one of them —
    # the 404 instead lists every suffixed name so the caller can pick one
    # explicitly. Deliberately no `hosted_by` here: naming one member would
    # be exactly the silent pick this exists to avoid.
    if suffixed_names:
        error["suffixed_lanes"] = list(suffixed_names)
        error["message"] += (
            " It is hosted by mesh members whose fingerprints disagree — "
            f"address one directly: {', '.join(suffixed_names)}."
        )
    return json.dumps({"error": error}).encode("utf-8")


def _role_unverified_body(
    requested: str | None,
    backend_name: str,
    pending_origin: str,
) -> bytes:
    """The boot-window **503** body — a "not yet", never a "never".

    Cloned deliberately from :func:`_role_infeasible_body` (same OpenAI error
    shape, same ``hosted_by`` referral key) with ONE difference that is the
    whole point: the ``type``/``code`` is ``role_unverified`` and the status
    is a retryable 503 rather than a terminal 404. The mesh member at
    ``pending_origin`` PUBLICLY ANNOUNCED this role but its first
    ``/capabilities`` probe has not landed yet — ``verified_roles == ()``
    alone conflated that boot window with "probed and verified nothing", and
    a caller that arrived during it was told the role would never be served
    here. A member that WAS probed and verified nothing is a hard negative
    and still gets the 404.
    """
    label = requested or "(unspecified)"
    return json.dumps(
        {
            "error": {
                "message": (
                    f"The model `{label}` is not served on this machine — the "
                    f"role it resolves to here (`{backend_name}`) is announced by "
                    f"the mesh member at `{pending_origin}`, which has not been "
                    "verified yet (its first /capabilities probe has not landed). "
                    "Retry shortly."
                ),
                "type": "role_unverified",
                "code": "role_unverified",
                "hosted_by": pending_origin,
            }
        }
    ).encode("utf-8")


def _role_unverified_response(
    mesh_snapshot: "RoutingSnapshot | None",
    role: str,
    placement: "RolePlacement",
    requested: str | None,
    backend_name: str,
) -> GatewayResponse | None:
    """The 503 ``role_unverified`` for a role whose only candidate is pending,
    or ``None`` when this is not the boot window.

    Extracted as ONE helper (rather than inlined at each of the three
    fall-through sites in :func:`handle_post`) so those already-branchy
    functions gain no cognitive complexity (Sonar S3776) and the three sites
    cannot drift apart on status, body or headers.

    Self-guarding, so a caller can invoke it unconditionally: it returns
    ``None`` whenever the role has a routable plain origin (forward there
    instead) or nothing pending (fall through to the terminal 404).
    """
    if placement.plain_origins or not placement.pending_origins:
        return None
    origin = placement.pending_origins[0]
    member_name = origin
    if mesh_snapshot is not None:
        for m in mesh_snapshot.members:
            if m.origin == origin:
                member_name = m.name
                break
    markers: list[tuple[str, str]] = []
    if mesh_snapshot is not None:
        markers = mesh_markers(mesh_snapshot, role, chosen_origin=origin, unverified=True)
    return GatewayResponse(
        status=503,
        headers=[
            ("Content-Type", _CONTENT_TYPE_JSON),
            ("Retry-After", str(BACKEND_UNAVAILABLE_RETRY_AFTER_SECONDS)),
        ]
        + markers
        + [(MESH_MEMBER_HEADER, member_name)],
        body=_role_unverified_body(requested, backend_name, origin),
    )


def _busy_body(requested_tier: str) -> bytes:
    """Return the JSON body for a 429 busy (shed) response."""
    _LANE_LABELS = {"main": "cortex", "multimodal": "senses", "muse": "muse"}
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


# --- the proxy data plane: follow the referral (proxy-lobes t6, #115/#127) --

# The THIRD lobe state — awake / asleep / PROXY. A role this box dropped
# (``table.infeasible``) that is opted in to proxying (``table.peer_proxied``,
# t1 — RETIRED SOURCE, t14: this used to be armed by a per-backend env knob;
# the mesh RoutingSnapshot is the candidate/forward source now) is answered by
# FORWARDING the request to that peer instead of the referral 404. The forward
# reuses the existing relay machinery unchanged (a synthetic Backend whose
# base_url is the operator-declared origin → open_upstream → buffered JSON or
# SSE chunk relay), plus four proxy-specific rules:
#
# * **pairwise credentials** — the outbound request carries ``Authorization:
#   Bearer <table.peer_api_keys[name]>`` when a per-peer key is declared, and
#   NO Authorization header otherwise. The CALLER's own credential is REMOVED
#   either way: it authenticated the caller to THIS box (t2's inbound gate)
#   and must never travel to a peer that was issued its own pairwise key.
# * **single hop** — every proxied departure is stamped with
#   ``X-Lobes-Proxied: <backend name>`` (an origin-less, key-free token). A
#   request that ARRIVES already carrying that marker and would depart again
#   via this branch is REFUSED (508 ``proxy_loop``, naming both hops) with
#   zero outbound attempts; a marked arrival whose role is served locally
#   processes normally (the marker only gates the proxy branch).
# * **response marker** — every response produced by this branch (a 2xx/4xx
#   relay, the peer-declined 404, the peer-down 503) carries
#   ``X-Lobes-Proxied-By: <peer origin, verbatim>`` so a caller can always
#   tell a proxied answer from a locally-served one (which NEVER carries it).
#   The loop refusal does not: nothing was proxied.
# * **local pressure bypass** — a proxied request skips this box's
#   swap/iowait tier shedding entirely: pressure describes THIS box's load,
#   and the model runs on the PEER, whose own gateway applies its own policy
#   when the forward arrives; shedding here too would double-gate the role on
#   the wrong box's load (the peer's 429 rides back through the 4xx relay).
#
# Failure modes mirror the single-owner rules (#91 — never a cross-model
# fallback): peer refused/timeout/>=500 → the retryable 503
# ``backend_unavailable`` + Retry-After; peer 2xx/4xx relays verbatim — with
# ONE exception: a peer 404 whose body is ``role_infeasible`` means the peer
# ALSO dropped the role (a misdeclared referral) → relay it terminally with
# the message rewritten to name the declining peer, and never attempt another
# hop.

PROXIED_HEADER = "X-Lobes-Proxied"
PROXIED_BY_HEADER = "X-Lobes-Proxied-By"

# --- replica-pool markers (cortex-replica-pool t7, issue #199) --------------
#
# Once a role can be served by more than one replica, "which box answered?"
# stops being inferable from "which gateway did I dial?". Before the pool the
# only marker was PROXIED_BY_HEADER, present exactly when the answer came from
# somewhere else — so its ABSENCE meant "served here". That inference dies with
# the pool for a caller who cannot see the config, so a pooled LOCAL answer now
# says so explicitly.
#
# * ``X-Lobes-Served-By`` — this box's OWN operator-declared origin
#   (``GATEWAY_SELF_ORIGIN`` → ``RoutingTable.self_origin``), or the literal
#   ``"local"`` when undeclared. It is never derived from the box's own view of
#   its network: the #92 lesson (origins are typed, never discovered) applies to
#   a box's self-name exactly as to a peer's.
# * ``X-Lobes-Route-Reason`` — WHY that replica was chosen, from the closed
#   vocabulary in :mod:`lobes.gateway._selection` (local-idle |
#   peer-less-loaded | local-busy-forwarded | affinity | sole-ready | none), so
#   a trace can tell a deliberate forward from a fallback without gateway logs.
# * ``X-Lobes-Affinity`` — an INBOUND, optional caller hint (a session/thread
#   key) that makes selection sticky to one replica while it stays selectable.
#   It is forwarded verbatim on a hop so the receiving gateway would make the
#   same sticky choice.
#
# Both response markers appear ONLY on a pooled role's answers (h1: a
# deployment with no ``*_PEER_ORIGINS`` and no mesh-verified member for the
# role is byte-identical to the pre-pool release, headers included).
SERVED_BY_HEADER = "X-Lobes-Served-By"
ROUTE_REASON_HEADER = "X-Lobes-Route-Reason"
AFFINITY_HEADER = "X-Lobes-Affinity"
# * ``X-Lobes-Route-Attempts`` — how many replicas this request was actually
#   dispatched to (t8, spec c15/h12). Present ONLY when it is greater than 1,
#   so the common single-attempt answer keeps exactly the t7 header set and a
#   trace that sees the header knows a pre-dispatch failure was survived.
ROUTE_ATTEMPTS_HEADER = "X-Lobes-Route-Attempts"
# * ``X-Lobes-Route-Load`` — the capacity and utilisation the placement
#   ACTUALLY used (capacity-relative pool routing, t5). Once ranking divides
#   active requests by a per-device capacity, "why did this land here?" stops
#   being answerable from the reason alone: ``peer-less-loaded`` says a peer
#   was less loaded *relative to its own capacity* and nothing says what that
#   capacity was. This is a NEW header rather than a new reason value on
#   purpose — :mod:`lobes.gateway._selection`'s reason vocabulary is closed
#   and t3 already redefined ``peer-less-loaded`` once; widening it again
#   would break a caller parsing the documented set. Format is a
#   semicolon-separated field list, stable and cheap to parse:
#
#       X-Lobes-Route-Load: active=1; capacity=8; utilisation=0.125; calibrated=true
#
#   ``capacity`` is the capacity RANKING used — for an uncalibrated replica
#   that is the neutral substitute, not the ``1.0`` sentinel read as one slot,
#   which is exactly what ``calibrated=false`` says out loud. Like every other
#   pool marker it rides ONLY on a pooled answer (h1).
ROUTE_LOAD_HEADER = "X-Lobes-Route-Load"

# The snapshot seam: role/backend name → that role's replicas as of the last
# background probe, local first. Injected into :func:`handle_post` exactly as
# ``open_upstream`` is, so dispatch is unit-testable with no sockets, no clock
# and no cache. ``None`` (every pre-pool call site) disables the pool path
# outright. :meth:`lobes.gateway._replicas.ReplicaCache.current` satisfies it —
# binding one is t8's job.
ReplicaSnapshot = Callable[[str], "tuple[ReplicaState, ...]"]

# The in-flight seam (capacity-relative pool routing, t5): given the backend
# name and the ORIGIN a request is about to be dispatched to, count that
# dispatch and hand back the callable that releases it. Injected exactly as
# ``replica_snapshot`` is, so the whole accounting path unit-tests with no
# cache, no clock and no sockets; ``None`` (every pre-t5 call site) means the
# pool dispatches without counting, which is precisely the pre-t5 behaviour.
#
# Why a returned release rather than a symmetric ``end(origin)``: the counter
# is token-based (:meth:`lobes.gateway._replicas.ReplicaCache.end_dispatch`),
# so a closure over the token makes it structurally impossible for one
# request's release to decrement another's dispatch to the same origin.
DispatchCounter = Callable[[str, str], "Callable[[], None]"]

# The release for a call site that has no counter wired. Named rather than a
# lambda so a stack trace through it is readable.


def _no_release() -> None:
    """Release nothing — the unpooled/uncounted dispatch's completion hook."""


# The literal ``X-Lobes-Served-By`` value for a box whose operator declared no
# GATEWAY_SELF_ORIGIN: honest ("I served it") without inventing a hostname.
_SELF_ORIGIN_FALLBACK = "local"

# 508 Loop Detected — the refusal for a marked request that would re-proxy.
_PROXY_LOOP_STATUS = 508

# Backend name → the deployment env var carrying that role's served model id
# (the same ``<PREFIX>_SERVED_NAME`` convention build_config reads). Consulted
# by _peer_served_name for a proxied role whose backend is UNWIRED locally.
_PEER_SERVED_NAME_ENV: dict[str, str] = {
    "primary": "PRIMARY_SERVED_NAME",
    "multimodal": "MULTIMODAL_SERVED_NAME",
    "muse": "MUSE_SERVED_NAME",
    "worker": "WORKER_SERVED_NAME",
    "associate": "ASSOCIATE_SERVED_NAME",
    # d1 reversal (2026-08-20) — hand is proxyable now; see
    # _config.NEVER_PROXIED_BACKENDS.
    "hand": "HAND_SERVED_NAME",
    "embed": "EMBED_SERVED_NAME",
    "rerank": "RERANK_SERVED_NAME",
}

# Backend names that resolve their served id from a hardcoded constant rather
# than from :data:`_PEER_SERVED_NAME_ENV` above or :data:`_PEER_ROLE_HINT`
# below: the two audio sidecars and the innereye render tenant, none of which
# is a switchable catalog gear. See _peer_served_name's early return.
_FIXED_SIDECAR_BACKENDS: frozenset[str] = frozenset({"stt", "tts", "innereye"})

# Backend name → the catalog ``role_hint`` of its canonical model — the same
# fallback lobes.roles uses to NAME an unwired role's model.
#
# Unlike the retired PEER_ORIGIN/PEER_PROXY/PEER_API_KEY family (t14), this
# dict and :data:`_PEER_SERVED_NAME_ENV` above are NOT retired: they still
# resolve a served id for a mesh-pooled role's ``/v1/models`` advertisement
# (see :func:`pooled_backends`'s mesh branch and its call site around
# ``_peer_served_name`` below), independent of any env-declared peer. A role
# that resolves NO served name here is dropped by :func:`peer_specs_from_table`
# at its ``if not served_name`` guard, so the (now-dormant, env-only) proxy
# path goes silently inert — no peer probe, no ``/v1/models`` entry, no
# :func:`_proxied_owner` match. That is exactly how ``worker`` shipped in
# 0.54.6: wired through _config.py's three (now-deleted) peer dicts but
# missing from these two, so its proxy knob did nothing on a box that only
# REACHES worker (no ``WORKER_BASE_URL``, hence no wired Backend to resolve
# off).
# ``stt``/``tts``/``innereye`` are proxyable too but resolve via
# _peer_served_name's fixed-sidecar early return, not these tables.
# tests/test_gateway_proxy.py::test_every_proxyable_role_resolves_a_served_name
# is the standing guard.
_PEER_ROLE_HINT: dict[str, str] = {
    "primary": "primary",
    "multimodal": "multimodal",
    "muse": "muse",
    "worker": "worker",
    # `associate` owns its own catalog role_hint (issue #244, t2) — it no
    # longer resolves through worker's, so promoting/demoting the checkpoint
    # carrying role_hint="worker" cannot move this resolution. Mirrors
    # lobes.roles.ROLE_ROLE_HINT and catalog.BACKEND_ROLE_CATALOG_HINT.
    "associate": "associate",
    "hand": "hand",  # d1 reversal — paired with _PEER_SERVED_NAME_ENV above (the 0.54.6 lesson)
    "embed": "embedding",
    "rerank": "reranker",
}


def _peer_served_name(table: RoutingTable, name: str, env: Mapping[str, str]) -> str:
    """The model id this box forwards/advertises for proxied backend ``name``.

    **The source-of-truth decision for an UNWIRED proxied role** (a dropped
    lobe realistically has no ``*_BASE_URL``, so no :class:`Backend` exists in
    the table): resolution order is

    1. the WIRED backend's ``served_name`` when one exists (the
       wired-but-infeasible shape — thor-lobe's unconditionally-wired primary)
       — the table's own declaration outranks everything;
    2. the deployment env's ``<PREFIX>_SERVED_NAME`` — what the shape render /
       operator declared the role WOULD serve (same var build_config reads);
    3. the catalog canonical id for the role — the same source
       :mod:`lobes.roles` uses to name an unwired role's model.

    Whatever is resolved here is only ever ADVERTISED after the peer probe
    confirms the peer's own ``/v1/models`` lists exactly this id
    (:func:`lobes.gateway._readiness.probe_peer_ready`) — resolution picks the
    name; the probe supplies the honesty. A misresolved/misdeclared name
    simply never advertises ready, and a forward naming it surfaces the peer's
    own honest 404.
    """
    if name in _FIXED_SIDECAR_BACKENDS:
        # Fixed non-catalog tenants: the two audio sidecars (issue #129) and
        # the innereye ComfyUI render tenant (issue #82). None has a
        # SupportedModel entry, so neither _PEER_SERVED_NAME_ENV nor
        # _PEER_ROLE_HINT can resolve them — the id is the SAME constant
        # lobes.roles advertises on /capabilities (lazy import: matches
        # capabilities_payload's own deferred lobes.roles import below).
        from lobes.roles import _INNEREYE_MODEL, _STT_MODEL, _TTS_MODEL

        return {"stt": _STT_MODEL, "tts": _TTS_MODEL, "innereye": _INNEREYE_MODEL}[name]
    wired = next((b.served_name for b in table.backends if b.name == name), None)
    if wired:
        return wired
    from_env = (env.get(_PEER_SERVED_NAME_ENV.get(name, "")) or "").strip()
    if from_env:
        return from_env
    hint = _PEER_ROLE_HINT.get(name)
    return next((m.id for m in SUPPORTED_MODELS if m.role_hint == hint), "")


# --- the peer-only pool: placing a role this box does NOT host -------------
#
# The FIFTH thing that can happen to a model-routed POST, and the mirror of
# the pool branch further down. That one may forward a role this box HOSTS
# because a peer replica is better placed; this one places a role this box
# hosts NOWHERE, across the replicas that do. It has to run BEFORE the
# referral/proxy branch, because that branch forwards to a single declared
# peer origin and would consume every request before any placement could
# happen — the exact behaviour measured on the Orin on 2026-08-30, where all
# traffic pinned to one of two equally-good peers. (Historical: at the time
# that was the SINGULAR ``<PREFIX>_PEER_ORIGIN`` env family; t14 retired that
# family — the origin now comes from the mesh RoutingSnapshot instead, and
# the ordering requirement is unchanged.)
#
# It keeps every guard the singular branch has:
#
# * **One hop, always.** A request that ARRIVES marked `X-Lobes-Proxied` is
#   not placed; it falls through to `_proxied_owner`, which answers 508
#   `proxy_loop` exactly as before (c7/h4). A box with no local replica has
#   nothing to serve a marked arrival with, so 508 is still the honest answer.
# * **Never worse than today.** Nothing selectable — no ready peer, or peers
#   that disagree with each other — returns None and the request takes the
#   pre-change path: the singular forward, or the terminal 404
#   `role_infeasible` when no singular origin is declared (frame decision c24).
# * **Pressure is not re-applied.** This sits on the same side of the pressure
#   gate as the singular proxy branch, so a role this box cannot serve is
#   never shed on this box's own swap/iowait (c21) — the peer's gateway
#   applies its own policy, and its 429 rides back through the one-forward
#   rule.


def pooled_backends(
    table: RoutingTable,
    replica_snapshot: ReplicaSnapshot | None,
    *,
    mesh_snapshot: RoutingSnapshot | None = None,
) -> frozenset[str]:
    """Backend names this box PLACES across replicas instead of pinning to one.

    The single predicate for "is this role pooled here", shared by the request
    path (:func:`_peer_only_forward`) and the ``/v1/models`` advertisement, so
    the two can never disagree about which roles the pool is answering for —
    a box that places a role but does not list it, or lists one it would not
    place, is lying in one direction or the other.

    A name qualifies via EITHER of two independent sources (t13, W9b): the
    table-declared one — this box does not host it, ``table.replica_origins``
    names plural replicas for it, ``table.peer_origins`` has the singular
    origin the fall-through and ``hosted_by`` both need, and at least one
    declared replica is right now compatible and ready (self-healing — every
    peer going unready drops the entry again) — or the mesh one: this box
    does not host it (it is in ``table.infeasible``) AND the mesh's own
    :class:`~lobes.gateway._mesh_routing.RoutingSnapshot` has at least one
    member verified for the role. In a normal deployment the table-declared
    source is permanently empty: t14 deleted the ``<PREFIX>_PEER_ORIGIN(S)``
    env parsing that used to populate ``table.replica_origins``/
    ``peer_origins``, so ``build_config`` never fills them any more — that
    branch fires only for a hand-built :class:`RoutingTable` (tests, or a
    future non-env source). The mesh source is what actually pools a role
    today: it reads ``table.infeasible`` (a pure hardware/shape fact) and
    :func:`~lobes.gateway._mesh_routing.compute_role_placement`'s
    ``plain_origins`` — the fingerprint-agreement-filtered subset of
    ``verified_roles``, never the raw union (two members that each verify a
    role but DISAGREE with each other on its fingerprint must never be
    pooled under the one plain name — that ambiguity is exactly what the
    suffixed-lane naming, issue #237, exists to keep out of the ranked pool).
    Both sources are unioned: a hand-built table's replica origins still
    behave exactly as they did pre-mesh (t14 only deleted the env parsing
    that fed them, not this branch), and the mesh is a first-class, additive
    source rather than an overlay gated behind an env-only precondition.
    """
    if replica_snapshot is None and mesh_snapshot is None:
        return frozenset()

    def _mesh_roles() -> frozenset[str]:
        """Backend names THIS BOX LACKS that the mesh has verified members for."""
        from lobes.roles import BACKEND_ROLE

        result: list[str] = []
        for backend_name in table.infeasible:
            role = BACKEND_ROLE.get(backend_name, backend_name)
            local_fp = _local_backend_fingerprint(replica_snapshot, backend_name)
            placement = compute_role_placement(mesh_snapshot, role, local_fingerprint=local_fp)
            if placement.plain_origins:
                result.append(backend_name)
        return frozenset(result)

    if replica_snapshot is not None:
        original = frozenset(
            name
            for name in table.replica_origins
            if name in table.infeasible
            and table.peer_origins.get(name)
            and any(s.compatible and s.ready and not s.local for s in replica_snapshot(name))
        )
    else:
        original = frozenset()
    if mesh_snapshot is None:
        return original

    # Mesh-augmented: roles the mesh verified even though no local replica
    # is ready and no plural/singular env origins are declared at all.
    mesh = _mesh_roles()
    return original | mesh


def _mesh_member_marker(
    mesh_snapshot: RoutingSnapshot | None, origin: str
) -> list[tuple[str, str]]:
    """``X-Lobes-Mesh-Member`` header for the mesh member serving ``origin``.

    Extracted from :func:`_peer_only_forward` (S3776) — same behaviour, just
    named: an origin the mesh snapshot does not recognise (or no snapshot at
    all) yields no marker, exactly as the inline loop did.
    """
    if mesh_snapshot is None:
        return []
    for m in mesh_snapshot.members:
        if m.origin == origin:
            return [(MESH_MEMBER_HEADER, m.name)]
    return []


def _peer_only_forward(
    table: RoutingTable,
    cfg: ServerConfig,
    peer_specs: Mapping[str, PeerSpec] | None,
    path: str,
    req_headers: list[tuple[str, str]],
    body: bytes,
    open_upstream: OpenUpstream,
    *,
    requested: str | None,
    replica_snapshot: ReplicaSnapshot | None,
    mesh_snapshot: RoutingSnapshot | None = None,
    counter: DispatchCounter | None = None,
) -> GatewayResponse | None:
    """Place one request across the replicas of a role this box does not host.

    ``None`` means "not applicable, take the pre-change path" — no snapshot,
    no plural origins, the role is hosted here, the request already crossed a
    hop, or nothing was selectable. Every one of those is a fall-through, not
    an error: the pool is an optimisation over the singular forward and can
    never leave a box worse off than it was without it.
    """
    backend_name = infeasible_owner(table, requested)
    # `pooled_backends` carries every precondition — dropped here, plural
    # origins, the singular origin the fall-through needs, and a compatible
    # ready replica (mesh-augmented, W9). A name it omits takes the
    # pre-change path untouched.
    if backend_name is None or backend_name not in pooled_backends(
        table, replica_snapshot, mesh_snapshot=mesh_snapshot
    ):
        return None
    if _arriving_hop_marker(req_headers) is not None:
        return None  # single hop: let _proxied_owner answer 508 as it always has
    placement = _pool_selection(
        table,
        backend_name,
        req_headers,
        replica_snapshot=replica_snapshot,
        mesh_snapshot=mesh_snapshot,
        # There is no local replica to be busy: `local_busy` only ever excludes
        # a LOCAL candidate (`_selection._is_selectable`), so False is not an
        # assumption about this box's load — it is the absence of a local
        # candidate to apply it to.
        local_busy=False,
    )
    origin = placement.selection.origin if placement is not None else None
    if not origin:
        return None
    spec = (peer_specs or {}).get(backend_name)
    # Name the serving MESH member (t13, mirrors `_pool_marker_headers`'s
    # non-local branch) — an origin the mesh snapshot knows about that env
    # never declared a peer for still gets `X-Lobes-Mesh-Member`, so a caller
    # can tell a mesh-sourced forward from an env-peer one exactly like the
    # hosted-pool path already can.
    member_marker = _mesh_member_marker(mesh_snapshot, origin)
    markers = (
        [(ROUTE_REASON_HEADER, placement.selection.reason)]
        + member_marker
        + _route_load_header(placement)
    )
    target = _ForwardTarget(
        name=backend_name,
        # The served id a peer is asked for is the SAME one the singular
        # forward would have used, so a caller cannot tell the two paths
        # apart by what the peer was asked to serve. With no PeerSpec (a
        # pooled-but-unproxied role) the body is forwarded verbatim below.
        origin=origin,
        served_name=spec.served_name if spec is not None else (requested or ""),
        api_key=_pool_replica_api_key(table, backend_name, origin, mesh_snapshot),
    )
    # COUNT the dispatch, exactly as `_pool_attempt` does for a hosted pool.
    # Probed load is up to one refresh interval stale, so without this every
    # concurrent arrival reads the same idle snapshot, ranks the same replica
    # first (ties break on origin string) and stampedes it — measured live on
    # the Orin on 2026-08-30, where four concurrent requests all placed onto
    # the same peer while the other sat idle. The counter is what makes the
    # snapshot self-correct BETWEEN refreshes; a peer-only pool needs it more
    # than a hosted one, because it has no local replica to absorb a tie.
    release = (counter or _uncounted)(backend_name, origin)
    answer: GatewayResponse | None = None
    try:
        answer = _proxy_to_peer(
            cfg,
            target,
            path,
            req_headers,
            body,
            open_upstream,
            rewrite=spec is not None,
            extra_response_headers=markers,
        )
        return _hand_off_release(answer, release)
    finally:
        # Same discipline as `_pool_attempt`: an answer still holding an
        # upstream is released by the handler after the relay; everything
        # else — a buffered body, the peer-down 503, an exception — is
        # complete here and released here. `end_dispatch` is idempotent, so a
        # double release is a no-op rather than a negative count.
        if answer is None or answer.on_complete is not release:
            release()


def peer_specs_from_table(
    table: RoutingTable, env: Mapping[str, str] | None = None
) -> dict[str, PeerSpec]:
    """One :class:`PeerSpec` per proxied role, from the routing table's config.

    The single builder both consumers share: :func:`serve` hands the specs to
    the :class:`ReadinessCache` (the peer-probe thread) AND to the handler
    (the data-plane branch in :func:`handle_post` + the ``/v1/models``
    advertisement), so the origin/served-id/key a probe verified are exactly
    the ones a forward dials. ``env`` defaults to ``os.environ`` (the same
    environment ``build_config`` built ``table`` from in the gateway
    container); see :func:`_peer_served_name` for the served-id resolution.
    Every name in ``table.peer_proxied`` has a declared origin by construction
    (:func:`lobes.gateway._config._peer_proxied` gates on it); the guard here
    only protects against a hand-built table violating that invariant. Key
    material rides only the ``repr``-hidden ``PeerSpec.api_key`` field.
    """
    # deferred import — see the module-level NOTE
    from lobes.roles import BACKEND_ROLE

    resolved_env = os.environ if env is None else env
    specs: dict[str, PeerSpec] = {}
    for name in sorted(table.peer_proxied):
        origin = table.peer_origins.get(name)
        if not origin:
            continue  # impossible via build_config; hand-built tables degrade safely
        served_name = _peer_served_name(table, name, resolved_env)
        if not served_name:
            # No honest model id resolved (unwired role, no <PREFIX>_SERVED_NAME,
            # no catalog hint — only reachable via a hand-built table naming a
            # role outside the core four). A blank id must never advertise,
            # probe, or match a request's blank/unspecified model in
            # _proxied_owner — so build no spec at all: the role degrades to
            # the referral-only 404, exactly as if the proxy knob were unset.
            continue
        specs[name] = PeerSpec(
            name=name,
            origin=origin,
            served_name=served_name,
            api_key=table.peer_api_keys.get(name),
            # A peer's GET /capabilities is keyed by ROLE, not backend (#220):
            # backend `multimodal` is role `senses`, `primary` is `cortex`. Fall
            # back to the backend name for a hand-built table naming something
            # outside the registry — PeerSpec.role_name() does the same.
            role=BACKEND_ROLE.get(name, name),
        )
    return specs


def _proxied_owner(
    table: RoutingTable, peer_specs: Mapping[str, PeerSpec], requested: str | None
) -> str | None:
    """The proxied backend name ``requested`` resolves to, else ``None``.

    Resolution mirrors — and slots between — the existing precedence rules in
    :func:`handle_post`:

    * the proxied role's own served id matches FIRST: an UNWIRED dropped
      role's id is in no wired backend and no alias, so
      :func:`infeasible_owner` cannot see it — but it IS advertised on
      ``/v1/models`` while the peer is ready, so it must forward, not 404;
    * a genuinely unknown id (h23) stays ``model_not_found`` — checked BEFORE
      the ``infeasible_owner`` fall-through below, because that helper routes
      unknown ids to ``default_model`` (whose owner may be the proxied role,
      e.g. thor-lobe's dropped cortex) and would otherwise silently forward a
      never-advertised id to the peer under the default model's identity;
    * everything else (role/tier aliases, wired-but-infeasible served ids, an
      UNSPECIFIED model routing to a proxied default) resolves through the
      same :func:`infeasible_owner` the referral 404 uses — the proxy branch
      replaces that 404 for exactly the names in ``table.peer_proxied``.
    """
    if not table.peer_proxied or not peer_specs:
        return None
    for name, spec in peer_specs.items():
        if name in table.peer_proxied and spec.served_name == requested:
            return name
    if is_unknown_model(table, requested):
        return None
    owner = infeasible_owner(table, requested)
    if owner is not None and owner in table.peer_proxied and owner in peer_specs:
        return owner
    return None


def _request_header(req_headers: Iterable[tuple[str, str]], name: str) -> str | None:
    """The first inbound header value matching ``name``, case-insensitively."""
    wanted = name.lower()
    for key, value in req_headers:
        if key.lower() == wanted:
            return value
    return None


def _arriving_hop_marker(req_headers: Iterable[tuple[str, str]]) -> str | None:
    """The inbound ``X-Lobes-Proxied`` marker value, if the request carries one."""
    return _request_header(req_headers, PROXIED_HEADER)


@dataclass(frozen=True)
class _ForwardTarget:
    """Where one outbound gateway→gateway forward is going, and as whom.

    Introduced by the replica pool (t7, issue #199) so :func:`_proxy_to_peer`
    stops being coupled to :class:`~lobes.gateway._readiness.PeerSpec`. The two
    callers now supply the same four facts from different config channels:

    * the REFERRAL/PROXY branch (issues #115/#127) builds one from a
      ``PeerSpec`` — the singular ``<PREFIX>_PEER_ORIGIN`` naming the box that
      hosts a role THIS box dropped;
    * the POOL branch builds one from the plural ``<PREFIX>_PEER_ORIGINS`` /
      ``<PREFIX>_PEER_API_KEYS`` pair — one of N interchangeable replicas of a
      role this box DOES host.

    Same wire contract either way (credential swap, single-hop marker,
    ``X-Lobes-Proxied-By`` attribution), which is the point of sharing the
    helper rather than growing a second forwarder that could drift out of
    agreement with it. ``api_key`` is ``repr=False``: the empty slot is legal
    (h29 — a peer with no inbound gate) and a non-empty one is a SECRET that
    must never reach a log, traceback or ``--json`` dump.
    """

    name: str
    origin: str
    served_name: str
    api_key: str = field(default="", repr=False)

    @classmethod
    def from_spec(cls, spec: PeerSpec) -> "_ForwardTarget":
        return cls(
            name=spec.name,
            origin=spec.origin,
            served_name=spec.served_name,
            api_key=spec.api_key or "",
        )


def _proxy_loop_body(arriving: str, spec: _ForwardTarget) -> bytes:
    """The 508 ``proxy_loop`` refusal body — names BOTH hops: the one already
    taken (the arriving marker value, stamped by the gateway that forwarded
    this request) and the one refused (this box's declared peer origin for the
    role). Never any key material."""
    return json.dumps(
        {
            "error": {
                "message": (
                    "refusing to proxy: this request already crossed one lobes "
                    f"proxy hop (X-Lobes-Proxied: {arriving}); forwarding it again "
                    f"to `{spec.origin}` for role `{spec.name}` could loop — peer "
                    "proxying is single-hop only (issues #115/#127)."
                ),
                "type": "proxy_loop",
                "code": "proxy_loop",
                "hops": [arriving, spec.origin],
            }
        }
    ).encode("utf-8")


def _peer_unavailable_response(spec: _ForwardTarget, attempts: list[str]) -> GatewayResponse:
    """The retryable 503 for a refused/timed-out/5xx'ing peer — the same
    owner-down convention local backends get (#14/#91: the peer is the ONE
    place this model lives; never a cross-model fallback), with the proxied-by
    marker naming which peer failed."""
    return GatewayResponse(
        status=503,
        headers=[
            ("Retry-After", str(BACKEND_UNAVAILABLE_RETRY_AFTER_SECONDS)),
            ("Content-Type", _CONTENT_TYPE_JSON),
            (PROXIED_BY_HEADER, spec.origin),
        ],
        body=_error_body(
            f"the peer hosting this model (`{spec.origin}`) is unavailable — retry shortly",
            attempts,
            error_type="backend_unavailable",
        ),
        attempts=attempts,
        peer_unavailable=True,
    )


def _peer_declined_body(spec: _ForwardTarget, raw: bytes) -> bytes | None:
    """The terminal body for a peer that answered 404 ``role_infeasible``.

    That verdict means the PEER also dropped the role — the operator's
    referral/proxy origin is misdeclared — so the error is TERMINAL (never
    another hop; there is no third box to ask). The peer's own body is kept,
    with its message rewritten to make unmistakable that the DECLARED PEER
    declined, not this gateway. Returns ``None`` when ``raw`` is not a
    ``role_infeasible`` error (any other 404 is the peer's authoritative
    client-error verdict and relays verbatim)."""
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    error = data.get("error") if isinstance(data, dict) else None
    if not isinstance(error, dict):
        return None
    if "role_infeasible" not in (error.get("code"), error.get("type")):
        return None
    original = error.get("message")
    prefix = (
        f"the declared peer for this role (`{spec.origin}`) declined it as "
        "role_infeasible — the peer does not host this role either (a "
        "misdeclared referral/proxy origin); no further hop is attempted."
    )
    if isinstance(original, str) and original:
        error["message"] = f"{prefix} Peer said: {original}"
    else:
        error["message"] = prefix
    data["error"] = error
    return json.dumps(data).encode("utf-8")


def _proxy_to_peer(
    cfg: ServerConfig,
    spec: PeerSpec | _ForwardTarget,
    path: str,
    req_headers: Iterable[tuple[str, str]],
    body: bytes,
    open_upstream: OpenUpstream,
    *,
    rewrite: bool = True,
    extra_response_headers: Iterable[tuple[str, str]] = (),
) -> GatewayResponse:
    """Forward one request to a peer gateway and relay the outcome.

    Accepts a :class:`PeerSpec` (the referral/proxy branch's config channel)
    or a :class:`_ForwardTarget` (the replica pool's, t7/#199) — the forward
    itself is identical either way, which is why both share this one function
    rather than growing a second forwarder that could drift out of agreement
    with the credential-swap / single-hop / attribution rules.

    ``extra_response_headers`` (t7) rides on EVERY outcome this produces —
    relay, peer-declined 404, peer-down 503 — so the pool can attach
    ``X-Lobes-Route-Reason`` beside ``X-Lobes-Proxied-By`` without the
    reason going missing on the failure paths a trace most needs it on. The
    508 loop refusal is the one exception below: nothing was proxied, so it
    carries no proxy attribution at all.

    See the section comment above for the full contract. The synthetic
    :class:`Backend` (``peer:<name>`` at the operator-declared origin) lets
    the UNCHANGED :func:`open_upstream` + relay machinery carry the forward —
    buffered JSON and SSE streaming both work exactly as for a local backend.

    ``rewrite`` (issue #129): the model-routed lanes rewrite the body's
    ``model`` to the peer's served id (aliases resolved HERE must not leak a
    name the peer doesn't serve); the AUDIO lanes are path-routed and forward
    the body VERBATIM — multipart uploads and the caller's own TTS JSON must
    arrive untouched, and the peer's gateway routes by path exactly as this
    one did.
    """
    req_headers = list(req_headers)
    target = spec if isinstance(spec, _ForwardTarget) else _ForwardTarget.from_spec(spec)
    arriving = _arriving_hop_marker(req_headers)
    if arriving is not None:
        # Single-hop guard: this request was already forwarded once by a peer
        # gateway; departing again could ping-pong between misconfigured boxes
        # forever. Refuse with zero outbound attempts. (A marked arrival whose
        # role is served LOCALLY never reaches this function — see handle_post.)
        return GatewayResponse(
            status=_PROXY_LOOP_STATUS,
            headers=[("Content-Type", _CONTENT_TYPE_JSON)],
            body=_proxy_loop_body(arriving, target),
        )
    response = _relay_to_target(cfg, target, path, req_headers, body, open_upstream, rewrite)
    extra = list(extra_response_headers)
    if extra:
        response.headers = extra + response.headers
    return response


def _peer_5xx_response(
    up: "_Upstream",
    spec: _ForwardTarget,
    peer_backend: Backend,
    proxied_by: tuple[str, str],
) -> GatewayResponse:
    """Map a ``>= 500`` peer response, special-casing a 508 proxy_loop.

    Extracted from :func:`_relay_to_target` (Sonar S3776). This also retires
    a second, later ``if up.status == 508`` branch in that function: because
    508 satisfies ``>= 500`` and this branch always returns, that later
    branch could never execute — dead code from the moment both were added,
    not a behaviour change to remove it.
    """
    if up.status == 508:
        # Special-case 508: if the peer returned proxy_loop, relay it
        # verbatim rather than laundering it into a retryable 503.
        raw = up.read_all()
        up.close()
        try:
            err_data = json.loads(raw)
            if isinstance(err_data, dict) and err_data.get("error", {}).get("type") == "proxy_loop":
                return GatewayResponse(
                    status=508,
                    headers=[("Content-Type", _CONTENT_TYPE_JSON), proxied_by],
                    body=raw,
                )
        except (json.JSONDecodeError, TypeError):
            pass
        # Not a proxy_loop error — treat as generic 503.
        return _peer_unavailable_response(spec, [f"{peer_backend.name}: HTTP {up.status}"])
    attempts = [f"{peer_backend.name}: HTTP {up.status}"]
    up.close()
    return _peer_unavailable_response(spec, attempts)


def _relay_to_target(
    cfg: ServerConfig,
    spec: _ForwardTarget,
    path: str,
    req_headers: list[tuple[str, str]],
    body: bytes,
    open_upstream: OpenUpstream,
    rewrite: bool,
) -> GatewayResponse:
    """The forward itself, past the single-hop guard: credential swap, dial,
    and outcome mapping. Split out of :func:`_proxy_to_peer` only to keep that
    function's marker/guard bookkeeping legible (Sonar S3776)."""
    streaming = is_streaming(body)
    fwd_body = rewrite_model(body, spec.served_name) if rewrite else body
    # Credential swap: the caller's Authorization authenticated it to THIS box
    # (t2's inbound gate) and must be provably absent outbound; the pairwise
    # per-peer key — when declared — is the only credential that travels.
    fwd_headers = [(k, v) for k, v in filter_headers(req_headers) if k.lower() != "authorization"]
    if spec.api_key:
        fwd_headers.append(("Authorization", f"Bearer {spec.api_key}"))
    fwd_headers.append((PROXIED_HEADER, spec.name))
    peer_backend = Backend(
        name=f"peer:{spec.name}", base_url=spec.origin, served_name=spec.served_name
    )
    proxied_by = (PROXIED_BY_HEADER, spec.origin)
    try:
        up = open_upstream(
            peer_backend,
            path,
            fwd_body,
            fwd_headers,
            connect_timeout=cfg.connect_timeout,
            read_timeout=cfg.read_timeout,
        )
    except UpstreamError as exc:
        return _peer_unavailable_response(spec, [str(exc)])
    if up.status >= 500:
        return _peer_5xx_response(up, spec, peer_backend, proxied_by)
    if up.status == 404:
        # The one 4xx that must be INSPECTED (mirroring the strict-retry path's
        # read-the-body rationale): a role_infeasible 404 is a misdeclared
        # referral and needs the peer named; any other 404 relays verbatim.
        raw = up.read_all()
        up.close()
        declined = _peer_declined_body(spec, raw)
        if declined is not None:
            return GatewayResponse(
                status=404,
                headers=[("Content-Type", _CONTENT_TYPE_JSON), proxied_by],
                body=declined,
            )
        return GatewayResponse(
            status=404, headers=[proxied_by] + _strip_peer_pool_markers(up.headers), body=raw
        )
    # 2xx or any other 4xx: the peer's authoritative verdict, relayed exactly
    # like the single-owner rules relay a local backend's (#91) — including
    # the peer's own 429 pressure shed riding back to the caller.
    return GatewayResponse(
        status=up.status,
        headers=[proxied_by] + _strip_peer_pool_markers(up.headers),
        upstream=up,
        streaming=streaming,
    )


_PEER_POOL_MARKERS = frozenset(
    {SERVED_BY_HEADER.lower(), ROUTE_REASON_HEADER.lower(), ROUTE_LOAD_HEADER.lower()}
)


def _strip_peer_pool_markers(headers: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Drop the PEER's own placement markers from a relayed answer.

    A pooled peer stamps ``X-Lobes-Served-By`` / ``X-Lobes-Route-Reason`` /
    ``X-Lobes-Route-Load`` on
    every answer it serves locally; relayed verbatim they would ride back next
    to THIS box's markers and a caller would see two ``X-Lobes-Route-Reason``
    values on one response (seen live 2026-08-25, #199 t11). The forwarder's
    verdict is the honest one — ``X-Lobes-Proxied-By`` already names the
    serving replica — so the peer's copies are dropped; every other upstream
    header (tier markers included) relays unchanged.
    """
    return [(k, v) for k, v in headers if k.lower() not in _PEER_POOL_MARKERS]


def _mesh_referral_origin(mesh_snapshot: "RoutingSnapshot | None", backend_name: str) -> str | None:
    """The origin of a verified mesh member hosting ``backend_name``, or ``None``.

    The referral 404's honesty source (t14): the retired env peer family
    (``table.peer_origins``, always empty now — see the "Retired" comment on
    :data:`lobes.gateway._config.NEVER_PROXIED_BACKENDS`) used to be the only
    way ``hosted_by`` was ever populated. The mesh RoutingSnapshot (t13) is
    the replacement source — when at least one mesh member has VERIFIED this
    role (its own ``/capabilities`` fingerprint agrees with every other
    member that also verifies it, :func:`~lobes.gateway._mesh_routing.
    compute_role_placement`'s ``plain_origins``), the first such origin is
    the referral; with no mesh, or no verified member, ``None`` — the
    pre-mesh, pre-referral body, byte for byte.
    """
    if mesh_snapshot is None:
        return None
    from lobes.roles import BACKEND_ROLE

    role = BACKEND_ROLE.get(backend_name, backend_name)
    placement = compute_role_placement(mesh_snapshot, role)
    return placement.plain_origins[0] if placement.plain_origins else None


def _feasibility_response(
    table: RoutingTable,
    requested: str | None,
    mesh_snapshot: "RoutingSnapshot | None" = None,
) -> GatewayResponse | None:
    """404 ``role_infeasible`` iff ``requested``'s owning backend is declared
    hardware-infeasible by this deployment's per-machine profile (task t6);
    ``None`` when there is no such gate to apply. Shared by both the
    tier-alias and plain-id resolution paths in :func:`handle_post` so the
    feasibility gate — which outranks pressure-shedding and is never bypassed
    by ``X-Lobes-Override`` — is checked identically in both.
    """
    infeasible_name = infeasible_owner(table, requested)
    if infeasible_name is None:
        return None
    # Opt-in honest referral (mesh-brain t3): when a peer hosts this role,
    # the 404 names it — as an ANNOTATION only. The request is still
    # answered HERE, terminally; a referral-only role is never forwarded
    # (the proxy data plane, t6, only fires for names in
    # ``table.peer_proxied``, which handle_post routes to _proxy_to_peer
    # BEFORE this gate — so every 404 built here stays byte-identical to the
    # pre-proxy contract). The env peer family that used to be the only
    # source of this origin is retired (t14; ``table.peer_origins`` is
    # always empty now) — the mesh RoutingSnapshot (t13) is the source
    # instead, via :func:`_mesh_referral_origin`. No declaration and no
    # verified mesh member → the pre-referral body, byte for byte.
    peer_origin = table.peer_origins.get(infeasible_name) or _mesh_referral_origin(
        mesh_snapshot, infeasible_name
    )
    return GatewayResponse(
        status=404,
        headers=[("Content-Type", _CONTENT_TYPE_JSON)],
        body=_role_infeasible_body(requested, infeasible_name, peer_origin),
    )


# The pressure sample handed to :func:`resolve_tier_request` once this box has
# already decided (via :func:`decide`) that the request is NOT shed. It is a
# read-only empty dict: every key defaults to 0.0, so the tier resolves warm.
# Passing the live sample instead would re-apply the pre-d1 shed rule that
# :func:`_resolve_tier` just declined to apply. (`_pooled_busy_dispatch` uses
# the same `{}` trick for the same reason.)
_WARM_SAMPLE: dict[str, float] = {}


def _role_is_pooled(
    table: RoutingTable,
    requested: str | None,
    replica_snapshot: ReplicaSnapshot | None,
    mesh_snapshot: "RoutingSnapshot | None" = None,
) -> bool:
    """Is this tier-alias request POOLED — declared peers AND a live snapshot,
    OR a mesh-verified plain-exposed member (t13)?

    The one input deviation ``d1`` needs to decide whether a host-level
    pressure verdict may refuse this request. Both halves matter: a
    declared-but-unsnapshotted role (every pre-pool call shape) is not pooled,
    because nothing here can know whether any replica has room. The mesh half
    needs neither ``<PREFIX>_PEER_ORIGIN(S)`` nor a ``replica_snapshot`` —
    only a mesh member PLAIN-exposed for this role (fingerprint-agreement
    filtered, never a raw ``verified_roles`` union — see
    :func:`_pool_selection`), independent of the env source.

    Deviation ``d5`` narrowed this helper. It used to return the local
    replica's load and published capacity too, and :func:`_resolve_tier` fed
    them to :func:`decide` as ``engine_active`` / ``engine_capacity`` — which
    made engine SATURATION a shed signal. That is admission control, not the
    routing preference c5/h4 specified, and t10 measured the cost live: an
    8-way flood at capacity 2 per box served 4/8 through the pool against 8/8
    with the pool bypassed. Saturation now drives SELECTION only
    (:func:`lobes.gateway._selection.is_full`), so a fleet with no headroom
    queues on the local owner instead of refusing. The engine plumbing is gone
    rather than left inert.

    Reading the snapshot is a dict lookup; no socket is opened here.
    """
    # The served name the tier WOULD resolve to, computed through the same pure
    # function with the override flag set so it resolves regardless of the
    # verdict still being decided. Mirrors `_pooled_busy_dispatch`.
    served = resolve_tier_request(requested, _WARM_SAMPLE, True, table)["served_name"]
    ordered = order_backends(table, served) if served else []
    if not ordered:
        return False
    backend_name = ordered[0].name
    if replica_snapshot is not None and table.replica_origins.get(backend_name):
        return True
    if mesh_snapshot is None:
        return False
    from lobes.roles import BACKEND_ROLE

    role = BACKEND_ROLE.get(backend_name, backend_name)
    local_fp = _local_backend_fingerprint(replica_snapshot, backend_name)
    placement = compute_role_placement(mesh_snapshot, role, local_fingerprint=local_fp)
    return bool(placement.plain_origins)


def _resolve_tier(
    table: RoutingTable,
    requested: str | None,
    pressure: dict[str, float],
    override: bool,
    replica_snapshot: ReplicaSnapshot | None = None,
    mesh_snapshot: "RoutingSnapshot | None" = None,
) -> tuple[GatewayResponse | None, str | None, list[tuple[str, str]], bool]:
    """The tier-alias branch of :func:`handle_post`: hardware feasibility gate,
    then pressure-aware busy shedding (#85, narrowed by ``d1``), then the
    resolved served name.

    Returns ``(early_response, served, tier_headers, busy)``. When
    ``early_response`` is not ``None`` the caller must return it immediately
    without dialing any backend; ``served``/``tier_headers`` are only
    meaningful otherwise.

    ``busy`` is the pressure verdict for THIS request, surfaced (t7, #199) so
    the pool's ``local_busy`` input is the decision already taken rather than a
    second sample of a moving signal. t8 stops returning early on it and
    forwards a busy pooled request to a selectable peer instead (spec c7/h6).

    Deviation ``d1`` (capacity-relative pool routing) narrows what may set it.
    t3 decoupled the SENDING side — a peer's host pressure verdict no longer
    gates whether we select it — but the RECEIVING side still shed, so the
    chain was "box A forwards → box B refuses on its own iowait reading → the
    429 relays back": a wasted round trip and the same refusal. So the shed
    verdict is taken HERE, from :func:`decide` with the pooled flag, rather
    than left to :func:`resolve_tier_request`'s host-only view:

    * a POOLED role is not shed on host ``iowait`` alone;
    * ``swap`` (paging) still sheds, pooled or not;
    * an UNPOOLED role decides exactly as it did before ``d1`` — h1's
      byte-identity for a no-peers deployment is untouched.

    Deviation ``d5`` removed the third bullet ``d1`` originally had here — "a
    FULL local engine still sheds". Engine saturation is a ROUTING
    PREFERENCE, not admission control: :func:`lobes.gateway._selection.is_full`
    keeps a full replica out of the pool so a replica WITH room wins, and when
    no replica anywhere has room the request is dialled locally and queues in
    the engine's own waiting queue. Feeding it to :func:`decide` instead
    routed the third concurrent arrival down the busy path, which 429s when no
    peer is selectable; t10 measured 4/8 served through the pool against 8/8
    with the pool bypassed. ``decide``'s ``engine_active`` / ``engine_capacity``
    parameters and their ``shed_signal="engine"`` verdict are untouched as a
    pure-function contract — this call site simply no longer supplies them, so
    the signal is unreachable from the request path.

    The ``shed_signal`` naming which fact justified a shed is available from
    the same verdict for t5 to surface on a trace.
    """
    early = _feasibility_response(table, requested, mesh_snapshot)
    if early is not None:
        return early, None, [], False

    verdict = decide(
        pressure.get("swap_used_percent", 0.0),
        pressure.get("iowait_percent", 0.0),
        requested,
        pooled=_role_is_pooled(table, requested, replica_snapshot, mesh_snapshot),
    )
    # `X-Lobes-Override` outranks every load condition (never the feasibility
    # gate above); an overridden request is not shed, exactly as before.
    if verdict["shed"] and not override:
        busy_response = GatewayResponse(
            status=429,
            headers=[
                ("Retry-After", str(BUSY_RETRY_AFTER_SECONDS)),
                ("X-Lobes-Tier-Reason", "busy"),
                ("Content-Type", _CONTENT_TYPE_JSON),
            ],
            body=_busy_body(verdict["requested_tier"]),
        )
        return busy_response, None, [], True

    decision = resolve_tier_request(requested, _WARM_SAMPLE, override, table)
    served = decision["served_name"]
    tier_headers = [
        ("X-Lobes-Tier", decision["served_tier"]),
        ("X-Lobes-Tier-Reason", decision["reason"]),
    ]
    return None, served, tier_headers, False


def _resolve_plain_model(
    table: RoutingTable,
    requested: str | None,
    mesh_snapshot: "RoutingSnapshot | None" = None,
) -> tuple[GatewayResponse | None, str | None]:
    """The non-tier branch of :func:`handle_post`: unknown-id 404 (h23), then
    the hardware feasibility gate, then the resolved served name.

    Returns ``(early_response, served)``; when ``early_response`` is not
    ``None`` the caller must return it immediately.
    """
    if is_unknown_model(table, requested):
        response = GatewayResponse(
            status=404,
            headers=[("Content-Type", _CONTENT_TYPE_JSON)],
            body=_model_not_found_body(requested),
        )
        return response, None
    early = _feasibility_response(table, requested, mesh_snapshot)
    if early is not None:
        return early, None
    return None, resolve_model(table, requested)


# --- the replica pool: local-vs-peer dispatch (t7, issue #199) --------------
#
# The pool is the FOURTH thing that can happen to a model-routed POST, and it
# differs from the referral/proxy branch above in one decisive way: that branch
# replaces a 404 for a role this box does NOT host, while the pool may forward
# a role this box DOES host, because a declared peer replica of the same role
# is better placed. So it cannot live beside `_proxied_owner`; it sits after
# the model has resolved to its owning backend and BEFORE that backend is
# dialed.
#
# Three rules make it safe to bolt onto a one-owner router (#91: never a
# different model):
#
# * **Same role, same model, or nothing.** A peer only enters the candidate set
#   when its live-probed fingerprint matches the local lane's
#   (`_replicas.py`). #91 forbids answering from a different model — it says
#   nothing about answering from an identical one on another box.
# * **Alias and raw served id take the identical path** (c31). Selection runs
#   off the OWNING BACKEND NAME, which both `model=cortex` and
#   `model=unsloth/Qwen3.8-27B-NVFP4` resolve to — and every deployed consumer
#   pins the raw id (the 2026-07-31 audit), so an alias-only pool would never
#   see a real caller.
# * **One hop, always** (c4/h4). An arriving `X-Lobes-Proxied` request skips
#   SELECTION entirely and is served by the local replica — not "selected, and
#   the local one happened to win", which a loaded box would get wrong. A box
#   with no local replica is the pre-existing `_proxied_owner` case and still
#   answers 508 `proxy_loop`.
#
# With no `<PREFIX>_PEER_ORIGINS` declared AND no mesh-verified member for
# the role, `_pool_selection` returns None before touching the snapshot and
# not one byte of the response changes (h1).


def _replica_api_key(table: RoutingTable, backend_name: str, origin: str) -> str:
    """This box's outbound credential for one replica ``origin``, or ``""``.

    Positional against ``table.replica_origins[backend_name]`` — index *i* of
    the keys tuple belongs to origin *i*, the parity `_config._replica_api_keys`
    enforces at startup (a length mismatch is a config error there, never a
    silent shift onto the wrong replica here). An EMPTY slot is legal and means
    "this peer runs no inbound gate" (h29 — the Thor sets no
    ``GATEWAY_API_KEY`` today); the forward then sends no Authorization at all
    rather than a blank Bearer. Key material never leaves this function except
    into the ``repr``-hidden :attr:`_ForwardTarget.api_key`.
    """
    origins = table.replica_origins.get(backend_name, ())
    keys = table.replica_api_keys.get(backend_name, ())
    try:
        index = origins.index(origin)
    except ValueError:
        return ""
    if index < len(keys) and keys[index]:
        return keys[index]
    # INHERIT the singular credential (peer-only-replica-pools, c19). The two
    # channels parse independently — `_replica_api_keys` returns nothing at all
    # when <PREFIX>_PEER_API_KEYS is unset — so a deployment that has been
    # forwarding happily on the singular pair for months would, on adding
    # plural origins, start sending NO Authorization to the very same peer and
    # collect a 401 from a box it was already authenticated to. Inheritance is
    # scoped to the origin that IS the singular peer: any other replica needs
    # its own slot, and an empty slot still means "this peer runs no inbound
    # gate" (h29), never "borrow someone else's key".
    if origin == table.peer_origins.get(backend_name):
        return table.peer_api_keys.get(backend_name, "")
    return ""


def _pool_replica_api_key(
    table: RoutingTable,
    backend_name: str,
    origin: str,
    mesh_snapshot: "RoutingSnapshot | None",
) -> str:
    """Outbound credential for one pooled replica ``origin`` (t13).

    An env-declared replica keeps :func:`_replica_api_key`'s existing
    resolution (its own slot, or the inherited singular credential)
    unchanged. An origin :func:`_replica_api_key` has nothing for BECAUSE it
    is a MESH-sourced candidate — no ``<PREFIX>_PEER_ORIGIN(S)`` ever named it
    — signs with the mesh join key instead, the same credential every other
    mesh-forward code path in this module already uses (never the caller's
    own bearer, which is stripped before every forward). A mesh-disabled box,
    or an origin that is neither an env replica nor a known mesh member,
    still gets ``""`` — no Authorization at all, exactly the pre-mesh
    behaviour.
    """
    key = _replica_api_key(table, backend_name, origin)
    if key:
        return key
    if mesh_snapshot is None or not any(m.origin == origin for m in mesh_snapshot.members):
        return ""
    try:
        mesh_cfg = _build_mesh_config()
    except MeshConfigError:
        return ""
    return (mesh_cfg.key if mesh_cfg.enabled else "") or ""


def _merge_mesh_candidates(
    candidates: list["ReplicaState"],
    mesh_snapshot: "RoutingSnapshot | None",
    mesh_plain_origins: tuple[str, ...],
) -> list["ReplicaState"]:
    """W9a/t13: merge mesh candidates as zero-load, neutrally-ranked replicas.

    Extracted from :func:`_pool_selection` (Sonar S3776) — identical
    behaviour, just named: mesh candidates are injected only when mesh is
    enabled and the backend's role has PLAIN-exposed members.
    """
    if mesh_snapshot is None:
        return candidates
    for m in mesh_snapshot.members:
        if m.origin in mesh_plain_origins:
            candidates.append(
                ReplicaState(
                    origin=m.origin,
                    local=False,
                    ready=True,
                    busy=False,
                    health="ok",
                    running=0,
                    waiting=0,
                    fingerprint=None,
                    compatible=True,
                    reason="",
                    last_seen=0.0,
                    weight=8.0,
                    calibrated=True,
                )
            )
    return candidates


def _maybe_synth_local_candidate(
    candidates: list["ReplicaState"],
    mesh_plain_origins: tuple[str, ...],
    backend_name: str,
    table: RoutingTable,
    local_busy: bool,
) -> list["ReplicaState"]:
    """Synthesize the local candidate for a mesh-only pool (Sonar S3776).

    Extracted from :func:`_pool_selection` — identical behaviour. A box that
    HOSTS the role is a pool member too. Local replica states come from the
    replica caches, which only exist for env-declared pools — so on a
    mesh-only host the local lane was never a candidate and every request
    was forwarded to a peer with reason "sole-ready" (live Spark reranker,
    2026-09-12). `select_replica` gives the synthesized candidate the tie
    (local wins ties) and ranks it like any replica otherwise.
    """
    if (
        mesh_plain_origins
        and not any(getattr(c, "local", False) for c in candidates)
        and backend_name not in table.infeasible
        and any(b.name == backend_name for b in table.backends)
    ):
        candidates.append(
            ReplicaState(
                origin=table.self_origin or "local",
                local=True,
                ready=True,
                busy=local_busy,
                health="ok",
                running=0,
                waiting=0,
                fingerprint=None,
                compatible=True,
                reason="local lane (mesh pool)",
                last_seen=0.0,
                weight=8.0,
                calibrated=False,
            )
        )
    return candidates


def _pool_selection(
    table: RoutingTable,
    backend_name: str,
    req_headers: list[tuple[str, str]],
    *,
    replica_snapshot: ReplicaSnapshot | None,
    mesh_snapshot: RoutingSnapshot | None = None,
    local_busy: bool,
    exclude: Collection[str] = (),
) -> "_Placement | None":
    """Run the selection policy for one pooled backend, or ``None``.

    ``None`` means **this request is not pooled at all** — neither
    ``<PREFIX>_PEER_ORIGINS`` nor a mesh-verified member exists for the
    owning backend's role — and the caller must take the pre-pool path with
    no markers whatsoever (h1).

    ``exclude`` (t8, spec c15/h12) drops replica ORIGINS the current request
    has already dispatched to and lost pre-dispatch, so the retry re-runs the
    same deterministic policy over what is left rather than walking a
    snapshot-ordered list. That keeps "at most once per replica" a property of
    the candidate set rather than of the loop, and keeps the LOCAL replica an
    ordinary candidate (it is excluded by its own origin like any other).

    Reading the snapshot is a dict lookup: no socket is ever opened here (the
    probes run on :class:`~lobes.gateway._replicas.ReplicaCache`'s background
    threads), so a hung peer can never delay local dispatch.

    When *mesh_snapshot* is provided and mesh has PLAIN-exposed members for
    this backend's role (:func:`~lobes.gateway._mesh_routing.compute_role_placement`'s
    ``plain_origins`` — the fingerprint-agreement-filtered subset of
    ``verified_roles``; a member whose fingerprint disagrees with the
    reference is never merged in here, only reachable via its own suffixed
    lane), mesh candidates are merged into the candidate set so that
    ``select_replica`` ranks across both local replicas and mesh-verified
    peers — INDEPENDENT of whether ``<PREFIX>_PEER_ORIGINS`` is declared at
    all (t13): a mesh-only box (no env peer origins, no local ``replica_snapshot``
    provider) still reaches this merge, because the entry guard below no
    longer requires the env source to be present.
    """
    from lobes.roles import BACKEND_ROLE

    role = BACKEND_ROLE.get(backend_name, backend_name)
    mesh_plain_origins: tuple[str, ...] = ()
    if mesh_snapshot is not None:
        local_fp = _local_backend_fingerprint(replica_snapshot, backend_name)
        mesh_plain_origins = compute_role_placement(
            mesh_snapshot, role, local_fingerprint=local_fp
        ).plain_origins
    if not table.replica_origins.get(backend_name) and not mesh_plain_origins:
        return None
    if _arriving_hop_marker(req_headers) is not None:
        # Single hop (c4/h4): a request a peer already forwarded is served
        # HERE or refused — never selected again, or two mutually-loaded boxes
        # would ping-pong it. `sole-ready` is the honest reason: the local
        # replica is the only one this request may consider. This holds under
        # local pressure too (t8): a marked arrival that this box cannot serve
        # gets this box's OWN 429, which is the receiver applying its own
        # policy (#85) — never a second forward.
        return _Placement(Selection(None, True, REASON_SOLE_READY), ())
    affinity = (_request_header(req_headers, AFFINITY_HEADER) or "").strip()
    candidates = list(replica_snapshot(backend_name)) if replica_snapshot is not None else []
    candidates = _merge_mesh_candidates(candidates, mesh_snapshot, mesh_plain_origins)
    candidates = _maybe_synth_local_candidate(
        candidates, mesh_plain_origins, backend_name, table, local_busy
    )
    if exclude:
        candidates = [c for c in candidates if c.origin not in exclude]
    return _Placement(
        select_replica(
            candidates,
            affinity=affinity or None,
            local_busy=local_busy,
        ),
        candidates,
    )


@dataclass(frozen=True)
class _Placement:
    """A :class:`Selection` plus the candidate set it was made over (t5).

    The candidates travel with the verdict because the CAPACITY a placement
    used is a property of the whole set, not of the chosen replica alone: an
    uncalibrated replica ranks at the neutral capacity
    :func:`~lobes.gateway._selection.selection_capacity` derives from its
    peers. Recomputing it from a second snapshot read would report numbers
    that no decision was ever made on — the snapshot folds in-flight
    dispatches and moves under concurrency — so the set is carried, not
    re-fetched.
    """

    selection: Selection
    candidates: "tuple[ReplicaState, ...]"


def _route_load_header(placement: "_Placement") -> list[tuple[str, str]]:
    """``X-Lobes-Route-Load`` for the chosen replica, or nothing (t5, c24/h17).

    Emitted only when the placement actually chose a replica that is in the
    candidate set — a ``sole-ready`` marked arrival ranked nothing, and
    reporting a utilisation for a decision that was never made would be a
    fabricated trace.
    """
    origin = placement.selection.origin
    chosen = next((c for c in placement.candidates if c.origin == origin), None)
    if chosen is None:
        return []
    capacity = selection_capacity(chosen, placement.candidates)
    active = chosen.running + chosen.waiting
    return [
        (
            ROUTE_LOAD_HEADER,
            f"active={active:d}; capacity={capacity:g}; "
            f"utilisation={selection_wait(chosen, placement.candidates):g}; "
            f"calibrated={'true' if is_calibrated(chosen) else 'false'}",
        )
    ]


def _stamp_pool_headers(table: RoutingTable, reason: str) -> list[tuple[str, str]]:
    """The markers a LOCALLY-served pooled answer carries (c19/h14, c37/h30).

    ``X-Lobes-Served-By`` names this box by its operator-declared
    ``GATEWAY_SELF_ORIGIN``, falling back to the literal ``"local"`` — never a
    hostname derived from the box's own view of its network (#92). A FORWARDED
    answer gets ``X-Lobes-Proxied-By`` from :func:`_proxy_to_peer` instead, plus
    the same reason header.
    """
    return [
        (SERVED_BY_HEADER, table.self_origin or _SELF_ORIGIN_FALLBACK),
        (ROUTE_REASON_HEADER, reason),
    ]


def _local_mesh_member_marker() -> list[tuple[str, str]]:
    """``X-Lobes-Mesh-Member`` naming THIS box, when the mesh join is armed.

    Best-effort and mesh-scoped only (item B, t9): a mesh-disabled box (no
    ``LOBES_MESH_KEY``) or one with no declared ``LOBES_MESH_NAME`` gets no
    marker at all — never a fabricated name — and a malformed mesh config
    (:class:`~lobes.gateway._mesh_config.MeshConfigError`) is swallowed the
    same way every other on-demand ``_build_mesh_config()`` call site in this
    module already tolerates it, so a local pooled answer is never turned
    into an error by this purely cosmetic header.
    """
    try:
        mesh_cfg = _build_mesh_config()
    except MeshConfigError:
        return []
    if not mesh_cfg.enabled or not mesh_cfg.name:
        return []
    return [(MESH_MEMBER_HEADER, mesh_cfg.name)]


# --- the pooled dispatch loop (t8, issue #199) ------------------------------
#
# t7 placed a pooled request; t8 gives that placement its FAILURE semantics.
# Three rules, each of which a naive "just retry" loop would get wrong:
#
# * **Pre-dispatch only** (c15/h12). A replica that refused, timed out, or
#   answered 5xx *before any bytes came back* never served the request, so
#   trying the next selectable replica costs the caller nothing. A replica
#   that answered 2xx and then dropped mid-stream DID serve it — the relay is
#   a one-shot byte tunnel with no buffering, and re-issuing would
#   double-charge the model and duplicate tokens. So the retry keys on
#   :attr:`GatewayResponse.peer_unavailable`, which is set on exactly the
#   pre-dispatch outcome and nothing else.
# * **At most once per replica** (c35/h27). Each dispatched origin is excluded
#   from the next selection pass, so the loop cannot revisit one and cannot
#   outlive the candidate set. The LOCAL replica is an ordinary candidate here
#   — it is excluded by its own origin like any peer.
# * **At most ONE forward per request** (c35/h27). A peer's own 429/4xx is its
#   authoritative verdict under ITS pressure policy (#85 — pressure describes
#   the box that samples it), so it rides back through the existing relay and
#   is NEVER retried locally or re-forwarded. Two mutually-loaded boxes
#   therefore produce exactly one forward and one 429, never a ping-pong.


LocalDial = Callable[["list[tuple[str, str]]"], "tuple[GatewayResponse | None, list[str]]"]


@dataclass(frozen=True)
class _PoolFallthrough:
    """ "Pooled, but nothing was dispatched" — the caller decides what that means.

    The two callers disagree, honestly: the normal path dials its LOCAL owner
    (the pre-pool behaviour, stamped with ``reason``), while the busy path has
    already been told by the pressure policy that the local owner must not take
    this request, so it returns the existing 429 instead.
    """

    reason: str


def _attempts_header(dispatched: int) -> list[tuple[str, str]]:
    """``X-Lobes-Route-Attempts`` iff more than one replica was dispatched to."""
    return [(ROUTE_ATTEMPTS_HEADER, str(dispatched))] if dispatched > 1 else []


def _pool_exhausted_response(
    tier_headers: list[tuple[str, str]], attempts: list[str], dispatched: int
) -> GatewayResponse:
    """Every selectable replica failed PRE-DISPATCH → the retryable 503.

    Same class as the single-owner owner-down 503 (#14/#91) — a transient
    "come back shortly", never a terminal "no such model" — but its
    ``attempts`` list names EVERY replica that was tried and how each failed,
    which is the only place an operator can see that the pool was exercised
    and exhausted rather than never consulted.
    """
    return GatewayResponse(
        status=503,
        headers=tier_headers
        + [
            ("Retry-After", str(BACKEND_UNAVAILABLE_RETRY_AFTER_SECONDS)),
            ("Content-Type", _CONTENT_TYPE_JSON),
            (ROUTE_REASON_HEADER, REASON_NONE),
        ]
        + _attempts_header(dispatched),
        body=_error_body(
            "every replica of this model is unavailable — retry shortly",
            attempts,
            error_type="backend_unavailable",
        ),
        attempts=attempts,
    )


@dataclass(frozen=True)
class _RequestCtx:
    """The six facts every dispatch-chain function needs about ONE inbound
    request, invariant across the whole pool retry loop.

    Introduced (Sonar S107) to fold ``table``/``cfg``/``path``/``req_headers``/
    ``body``/``open_upstream`` — previously six separate leading positional
    parameters on :func:`_pool_dispatch`, :func:`_pool_attempt` and
    :func:`_dial_selected` — into one object, after ``mesh_snapshot`` (t7/t8)
    became each function's 14th parameter. Purely mechanical: every caller
    and callee stays inside this module.
    """

    table: RoutingTable
    cfg: ServerConfig
    path: str
    req_headers: list[tuple[str, str]]
    body: bytes
    open_upstream: OpenUpstream


def _pool_dispatch(
    ctx: _RequestCtx,
    *,
    backend_name: str,
    served: str,
    tier_headers: list[tuple[str, str]],
    replica_snapshot: ReplicaSnapshot | None,
    mesh_snapshot: RoutingSnapshot | None = None,
    local_busy: bool,
    dial_local: LocalDial | None,
    counter: DispatchCounter | None = None,
) -> GatewayResponse | _PoolFallthrough | None:
    """Place and dispatch one pooled request, retrying pre-dispatch failures.

    Returns ``None`` when the request is **not pooled at all** (no snapshot
    provider, or neither ``<PREFIX>_PEER_ORIGINS`` nor a mesh-verified member
    exists for ``backend_name``'s role — see :func:`_pool_selection`) — the
    caller must then take the pre-pool path with no markers whatsoever (h1).
    Returns :class:`_PoolFallthrough` when the role IS pooled but nothing was
    selectable and therefore nothing was dispatched. Otherwise returns the
    answer, from whichever replica produced it.

    See the section comment above for the three rules this loop encodes.
    """
    attempts: list[str] = []
    excluded: set[str] = set()
    dispatched = 0
    while True:
        placement = _pool_selection(
            ctx.table,
            backend_name,
            ctx.req_headers,
            replica_snapshot=replica_snapshot,
            mesh_snapshot=mesh_snapshot,
            local_busy=local_busy,
            exclude=excluded,
        )
        if placement is None:
            return None
        selection = placement.selection
        if selection.origin is None:
            break
        if selection.local and dial_local is None:
            # Defensive only: `select_replica` never returns the local replica
            # when `local_busy` is set, which is the one case that passes no
            # local dialer. Treat it as "nothing to dispatch" rather than
            # inventing a local dial the pressure policy just forbade.
            break
        dispatched += 1
        response, failure = _pool_attempt(
            ctx,
            backend_name=backend_name,
            served=served,
            placement=placement,
            tier_headers=tier_headers,
            dispatched=dispatched,
            dial_local=dial_local,
            counter=counter,
            mesh_snapshot=mesh_snapshot,
        )
        if response is not None:
            return response
        attempts.extend(failure)
        excluded.add(selection.origin)
    if dispatched == 0:
        return _PoolFallthrough(selection.reason)
    return _pool_exhausted_response(tier_headers, attempts, dispatched)


def _pool_attempt(
    ctx: _RequestCtx,
    *,
    backend_name: str,
    served: str,
    placement: "_Placement",
    tier_headers: list[tuple[str, str]],
    dispatched: int,
    dial_local: LocalDial | None,
    counter: DispatchCounter | None = None,
    mesh_snapshot: RoutingSnapshot | None = None,
) -> tuple[GatewayResponse | None, list[str]]:
    """One dispatch to one selected replica.

    ``(response, [])`` when the replica answered anything the caller is
    entitled to (2xx, or its own authoritative 4xx — including a peer's 429
    pressure shed, which is never retried); ``(None, failures)`` when it
    failed PRE-DISPATCH and the next selectable replica may be tried.
    Extracted from :func:`_pool_dispatch` purely to keep that loop legible
    (Sonar S3776).

    **This is the ONLY place a replica-pool dispatch is counted** (t5). The
    counter is taken BEFORE the dial and released on every way out of this
    function, with exactly one exception: an answer that still has an
    ``upstream`` to relay is not finished yet, so its release rides on the
    response (:meth:`GatewayResponse.release`) and the handler fires it after
    the relay. Both leaky shapes are closed here rather than at the call
    site: a PRE-DISPATCH failure releases before returning, so the retry that
    follows cannot leave the refusing replica counted, and a
    gateway-generated body (the owner-down 503, a relayed error with no
    upstream) releases immediately because nothing further will touch it.
    """
    selection = placement.selection
    origin = selection.origin or ""
    markers = _pool_marker_headers(ctx.table, placement, dispatched, mesh_snapshot=mesh_snapshot)
    release = (counter or _uncounted)(backend_name, origin)
    answer: GatewayResponse | None = None
    try:
        answer, failures = _dial_selected(
            ctx,
            backend_name=backend_name,
            served=served,
            selection=selection,
            markers=markers,
            tier_headers=tier_headers,
            dial_local=dial_local,
            mesh_snapshot=mesh_snapshot,
        )
        if answer is None:
            return None, [f"{origin}: {failure}" for failure in failures]
        return _hand_off_release(answer, release), []
    finally:
        # Whatever path was taken, a dispatch whose completion did NOT escape
        # this block is released right here: a pre-dispatch failure (so the
        # retry below cannot leave the refusing replica counted), an
        # exception, and any answer already complete in memory.
        # `_hand_off_release` moves the release onto the response ONLY for an
        # answer with an ``upstream`` still to relay, and clears it from this
        # frame when it does — and `end_dispatch` is idempotent regardless, so
        # a double release is a no-op rather than a negative count.
        if answer is None or answer.on_complete is not release:
            release()


def _dial_selected(
    ctx: _RequestCtx,
    *,
    backend_name: str,
    served: str,
    selection: Selection,
    markers: list[tuple[str, str]],
    tier_headers: list[tuple[str, str]],
    dial_local: LocalDial | None,
    mesh_snapshot: "RoutingSnapshot | None" = None,
) -> tuple[GatewayResponse | None, list[str]]:
    """Dial the one replica ``selection`` chose — local owner or peer forward.

    Split out of :func:`_pool_attempt` so that function is a pure
    count/dial/release sandwich with a single ``finally``: the leak guard is
    then readable in one screen instead of straddling two branches.
    """
    if selection.local:
        if dial_local is None:  # pragma: no cover - guarded by _pool_dispatch
            return None, []
        return dial_local(markers + tier_headers)
    forwarded = _proxy_to_peer(
        ctx.cfg,
        _ForwardTarget(
            name=backend_name,
            origin=selection.origin or "",
            served_name=served,
            api_key=_pool_replica_api_key(
                ctx.table, backend_name, selection.origin or "", mesh_snapshot
            ),
        ),
        ctx.path,
        ctx.req_headers,
        ctx.body,
        ctx.open_upstream,
        extra_response_headers=markers,
    )
    if not forwarded.peer_unavailable:
        return forwarded, []
    return None, forwarded.attempts


def _counted_local_dial(
    dial_local: LocalDial,
    headers: list[tuple[str, str]],
    release: "Callable[[], None]",
) -> "tuple[GatewayResponse | None, list[str]]":
    """``dial_local(headers)`` with *release* fired on every way out.

    The same count/dial/release sandwich :func:`_pool_attempt` uses, and for
    the same reason: an answer with an ``upstream`` still to relay is not
    finished, so its release rides on the response and the handler fires it
    after the relay; everything else (a buffered answer, the owner-down
    failure, an exception) is complete here and is released here.
    """
    answer: GatewayResponse | None = None
    try:
        answer, attempts = dial_local(headers)
        if answer is not None:
            answer = _hand_off_release(answer, release)
        return answer, attempts
    finally:
        if answer is None or answer.on_complete is not release:
            release()


def _hand_off_release(response: GatewayResponse, release: "Callable[[], None]") -> GatewayResponse:
    """Move ``release`` onto ``response`` iff its completion escapes the
    dispatch block — i.e. it still has an ``upstream`` for the handler to
    relay. A buffered/gateway-generated answer is already complete, so it
    keeps ``on_complete=None`` and :func:`_pool_attempt`'s ``finally``
    releases it there instead."""
    if response.upstream is not None:
        response.on_complete = release
    return response


def _uncounted(_backend_name: str, _origin: str) -> "Callable[[], None]":
    """The no-counter :data:`DispatchCounter`: counts nothing, releases nothing."""
    return _no_release


def _pool_marker_headers(
    table: RoutingTable,
    placement: "_Placement",
    dispatched: int,
    *,
    mesh_snapshot: "RoutingSnapshot | None" = None,
) -> list[tuple[str, str]]:
    """The pool markers for one attempt: served-by (local only) + reason +
    the capacity/utilisation the placement used (t5) + attempts.

    When the chosen replica is a non-local MESH member (t8, #237), also adds
    ``X-Lobes-Mesh-Member`` naming it — the same marker the mesh-forward and
    suffixed-lane paths stamp, so every pooled/proxied/suffixed mesh answer
    carries one consistent header regardless of which code path served it.

    A LOCAL pick is named too (item B, t9): when this box's own mesh join is
    enabled and has a declared ``LOBES_MESH_NAME``, that name is stamped here
    as well, so every mesh answer — local or forwarded — names its serving
    member, not just the forwarded ones. A mesh-disabled or unnamed box keeps
    the pre-existing markers unchanged (no ``MeshConfigError`` ever escapes
    this best-effort lookup).
    """
    selection = placement.selection
    load = _route_load_header(placement)
    if selection.local:
        base = _stamp_pool_headers(table, selection.reason) + load + _attempts_header(dispatched)
        return base + _local_mesh_member_marker()
    member_marker: list[tuple[str, str]] = []
    if mesh_snapshot is not None and selection.origin:
        for m in mesh_snapshot.members:
            if m.origin == selection.origin:
                member_marker = [(MESH_MEMBER_HEADER, m.name)]
                break
    return (
        [(ROUTE_REASON_HEADER, selection.reason)]
        + member_marker
        + load
        + _attempts_header(dispatched)
    )


def _pooled_busy_dispatch(
    table: RoutingTable,
    cfg: ServerConfig,
    path: str,
    req_headers: list[tuple[str, str]],
    body: bytes,
    open_upstream: OpenUpstream,
    *,
    requested: str,
    replica_snapshot: ReplicaSnapshot | None,
    mesh_snapshot: RoutingSnapshot | None = None,
    busy_response: GatewayResponse,
    local_busy: bool,
    counter: DispatchCounter | None = None,
) -> GatewayResponse | None:
    """Under local pressure, forward a POOLED request instead of shedding it (c7/h6).

    #85 shed a `main`/`multimodal` request with 429 because this box was the
    only place the model lived. Once a role has replicas that premise is gone:
    the honest answer to "this box is swapping" is "the request goes to the box
    that is not", and the 429 is reserved for "no replica anywhere can take
    it". Pressure still describes THIS box only — the peer's own gateway
    applies its own policy, and its 429 (if it sheds too) rides straight back
    to the caller through the one-forward rule.

    Returns ``None`` when the request is not pooled, so the caller returns the
    pre-pool 429 byte-for-byte (h1). Otherwise: the forwarded answer, the
    exhausted-503, or ``busy_response`` with ``X-Lobes-Route-Reason`` prepended
    so a trace can tell "shed because nothing was free" from "shed because this
    box was never pooled".
    """
    # t13: no longer requires `replica_snapshot` (the env-sourced ReplicaCache
    # provider) to be present — a mesh-only box (no `<PREFIX>_PEER_ORIGINS`
    # anywhere) has no such cache at all, but `_pool_dispatch`/`_pool_selection`
    # below still find mesh candidates for the busy role via `mesh_snapshot`
    # and no-op safely (`replica_snapshot is None` → an empty local candidate
    # list) when neither source has anything.
    if not local_busy:
        return None
    # The served name the tier WOULD have resolved to. `_resolve_tier` returned
    # None for it (the busy short-circuit happens before resolution), so it is
    # recomputed here through the same pure decision function with the override
    # flag set — the one input that makes it resolve under pressure. This does
    # NOT honour `X-Lobes-Override` on the caller's behalf: an overridden
    # request is never busy in the first place, so it never reaches here.
    served = resolve_tier_request(requested, {}, True, table)["served_name"]
    ordered = order_backends(table, served) if served else []
    if not ordered:
        return None
    outcome = _pool_dispatch(
        _RequestCtx(table, cfg, path, req_headers, body, open_upstream),
        backend_name=ordered[0].name,
        served=served,
        tier_headers=[],
        replica_snapshot=replica_snapshot,
        mesh_snapshot=mesh_snapshot,
        local_busy=True,
        dial_local=None,
        counter=counter,
    )
    if outcome is None:
        return None
    if isinstance(outcome, _PoolFallthrough):
        busy_response.headers = [(ROUTE_REASON_HEADER, outcome.reason)] + busy_response.headers
        return busy_response
    return outcome


def _try_backends(
    ordered: list[Backend],
    cfg: ServerConfig,
    path: str,
    fwd_body: bytes,
    fwd_headers: list[tuple[str, str]],
    open_upstream: OpenUpstream,
    streaming: bool,
    tier_headers: list[tuple[str, str]],
) -> tuple[GatewayResponse | None, list[str]]:
    """Attempt each backend in ``ordered`` (in practice exactly one — no
    cross-backend failover, #91) and relay the first 2xx/4xx verbatim.

    Returns ``(response, attempts)``: ``response`` is ``None`` iff every
    backend refused / timed out / 5xx'd, in which case the caller maps
    ``attempts`` to the retryable 503.
    """
    attempts: list[str] = []
    for backend in ordered:
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
        return (
            GatewayResponse(
                status=up.status,
                headers=tier_headers + up.headers,
                upstream=up,
                streaming=streaming,
                attempts=attempts,
            ),
            attempts,
        )
    return None, attempts


def _try_backend_with_strict_retry(
    backend: Backend,
    cfg: ServerConfig,
    path: str,
    injected_body: bytes,
    original_body: bytes,
    tool_names: list[str],
    fwd_headers: list[tuple[str, str]],
    open_upstream: OpenUpstream,
    streaming: bool,
    tier_headers: list[tuple[str, str]],
) -> tuple[GatewayResponse | None, list[str]]:
    """The force-strict-tools dial (GATEWAY_FORCE_STRICT_TOOLS, opt-in): try
    ``backend`` once with ``injected_body``; on an HTTP 4xx/5xx whose body
    matches :data:`_STRICT_FAILURE_SIGNATURES`, retry EXACTLY ONCE with
    ``original_body`` (the un-injected request) — never a second retry.

    Deliberately bypasses :func:`_try_backends` only far enough to read the
    failure body: that function never reads a ``>=500`` body (it treats any
    5xx as owner-down and swallows it into the generic retryable 503 below),
    but a compile-failure signature can only be read by inspecting the body.
    The gateway's documented error contract is otherwise preserved on BOTH
    hops: a non-signature ``>=500`` (and a ``>=500`` on the retry) is still
    owner-down — attempt recorded, ``response=None``, caller maps it to the
    retryable 503 — while a non-signature 4xx is a genuine client error and
    relays verbatim. Only a signature-matching failure is treated as our own
    injection's fault and retried un-injected.

    A connect failure at either hop (initial or retry) degrades exactly like
    ``_try_backends`` — an attempt string appended, ``response=None`` — so
    the caller's existing owner-down 503 tail in :func:`handle_post` is
    unaffected either way.
    """
    attempts: list[str] = []
    try:
        up = open_upstream(
            backend,
            path,
            injected_body,
            fwd_headers,
            connect_timeout=cfg.connect_timeout,
            read_timeout=cfg.read_timeout,
        )
    except UpstreamError as exc:
        attempts.append(str(exc))
        return None, attempts
    if up.status < 400:
        return (
            GatewayResponse(
                status=up.status,
                headers=tier_headers + up.headers,
                upstream=up,
                streaming=streaming,
                attempts=attempts,
            ),
            attempts,
        )
    # A 4xx/5xx: read the FULL body (never done for a >=500 in _try_backends)
    # so the compile-failure signature can actually be checked.
    body_bytes = up.read_all()
    up.close()
    if not _matches_strict_failure_signature(body_bytes):
        if up.status >= 500:
            # Not our injection's fault and the owner is erroring: same
            # owner-down contract as _try_backends — record the attempt and
            # let the caller map it to the retryable 503.
            attempts.append(f"{backend.name}: HTTP {up.status}")
            return None, attempts
        # A non-signature 4xx is a genuine client error — relay verbatim.
        return (
            GatewayResponse(
                status=up.status,
                headers=tier_headers + up.headers,
                body=body_bytes,
                streaming=False,
                attempts=attempts,
            ),
            attempts,
        )
    _log_strict_retry(tool_names, body_bytes)
    try:
        retry_up = open_upstream(
            backend,
            path,
            original_body,
            fwd_headers,
            connect_timeout=cfg.connect_timeout,
            read_timeout=cfg.read_timeout,
        )
    except UpstreamError as exc:
        attempts.append(str(exc))
        return None, attempts
    if retry_up.status >= 500:
        # The un-injected retry also 5xx'd: that IS an owner-down condition —
        # same contract as _try_backends, mapped to the retryable 503.
        attempts.append(f"{backend.name}: HTTP {retry_up.status}")
        retry_up.close()
        return None, attempts
    return (
        GatewayResponse(
            status=retry_up.status,
            headers=tier_headers + retry_up.headers,
            upstream=retry_up,
            streaming=streaming,
            attempts=attempts,
        ),
        attempts,
    )


def _dial_owner(
    ordered: list[Backend],
    cfg: ServerConfig,
    path: str,
    fwd_body: bytes,
    fwd_headers: list[tuple[str, str]],
    open_upstream: OpenUpstream,
    streaming: bool,
    tier_headers: list[tuple[str, str]],
) -> tuple[GatewayResponse | None, list[str]]:
    """Dial the resolved owner once, via the strict-tools lane when armed.

    Force-strict-tools (opt-in, colleague#320): only a lane in
    :data:`_STRICT_TOOL_LANES` (currently the primary/cortex lane only — see
    that constant for why muse is excluded), only chat-completions, only a
    body an injection actually changed. Every
    other request takes the untouched :func:`_try_backends` call below — this
    is the byte-identical-passthrough guarantee when the knob is off (or
    simply inapplicable to this request). Extracted from :func:`handle_post`
    verbatim (Sonar S3776); returns exactly what the dial helpers return:
    ``(response-or-None, attempts)``.
    """
    strict_injection = None
    if (
        cfg.force_strict_tools
        and ordered[0].name in _STRICT_TOOL_LANES
        and _is_chat_completions_request(path)
    ):
        strict_injection = inject_strict_tools(fwd_body)

    if strict_injection is not None:
        injected_body, tool_names = strict_injection
        return _try_backend_with_strict_retry(
            ordered[0],
            cfg,
            path,
            injected_body,
            fwd_body,
            tool_names,
            fwd_headers,
            open_upstream,
            streaming,
            tier_headers,
        )
    return _try_backends(
        ordered,
        cfg,
        path,
        fwd_body,
        fwd_headers,
        open_upstream,
        streaming,
        tier_headers,
    )


def _resolve_served_or_early(
    table: RoutingTable,
    cfg: ServerConfig,
    path: str,
    req_headers: list[tuple[str, str]],
    body: bytes,
    open_upstream: OpenUpstream,
    *,
    requested: str,
    pressure: dict[str, float] | None,
    override: bool,
    replica_snapshot: ReplicaSnapshot | None,
    mesh_snapshot: RoutingSnapshot | None = None,
    dispatch_counter: DispatchCounter | None = None,
) -> tuple[GatewayResponse | None, str | None, list[tuple[str, str]], bool]:
    """Resolve ``requested`` to its served backend name, or a short-circuit.

    Extracted verbatim from :func:`handle_post` (Sonar S3776): this is the
    two branches that decide what gets dialed — the pressure-aware tier path
    (with its pooled-busy-forward carve-out) and the plain h23 unknown-model
    path. Returns ``(early, served, tier_headers, local_busy)``: when
    ``early`` is not ``None`` the caller must return it immediately without
    dialing anything.
    """
    tier_headers: list[tuple[str, str]] = []
    local_busy = False
    if pressure is not None and is_tier_alias(requested):
        # Hardware feasibility gate (issue #92 extended to the HARDWARE
        # dimension, task t6) runs BEFORE pressure-shedding/upward-fallback: an
        # infeasible role is an absolute hardware fact, not a load condition, so
        # it takes priority over — and is never bypassed by — X-Lobes-Override.
        # Checked on the LITERAL requested tier so an explicitly-named
        # infeasible role (e.g. "cortex") is rejected outright, never silently
        # re-routed to a different, feasible gear via the tier system's normal
        # upward-fallback substitution.
        early, served, tier_headers, local_busy = _resolve_tier(
            table, requested, pressure, override, replica_snapshot, mesh_snapshot
        )
        if early is not None:
            # t8 (spec c7/h6): a POOLED role that this box is too loaded to
            # serve is forwarded to a selectable peer replica instead of shed.
            # `local_busy` is the verdict already taken above, not a second
            # sample of a moving signal. Not pooled, or nothing selectable →
            # the pre-pool 429 (with the honest route reason when pooled).
            forwarded = _pooled_busy_dispatch(
                table,
                cfg,
                path,
                req_headers,
                body,
                open_upstream,
                requested=requested,
                replica_snapshot=replica_snapshot,
                mesh_snapshot=mesh_snapshot,
                busy_response=early,
                local_busy=local_busy,
                counter=dispatch_counter,
            )
            return (forwarded if forwarded is not None else early), None, tier_headers, local_busy
        return None, served, tier_headers, local_busy
    # h23 converse: an UNKNOWN non-empty id (never an alias, never a wired
    # backend's served name) must NOT be silently served under the default
    # backend's weights — reject it with a 404 model_not_found BEFORE routing,
    # matching what a real OpenAI/vLLM backend emits. Unknown-ness is decided
    # against the ROUTING TABLE (is_unknown_model), never the readiness-filtered
    # /v1/models list — so a wired-but-dead backend (dropped from /v1/models but
    # still in the table) is KNOWN and routes on to the retryable 503 below, not
    # a 404 (that distinction is what keeps issue #91 fixed). An UNSPECIFIED
    # (missing/blank) model is not unknown — it routes to default_model. The
    # hardware feasibility gate (task t6) mirrors the tier branch above: it
    # runs AFTER the unknown-model check (a genuinely never-advertised id
    # still gets model_not_found, not role_infeasible) but BEFORE
    # resolving/dialing a backend.
    early, served = _resolve_plain_model(table, requested, mesh_snapshot)
    return early, served, tier_headers, local_busy


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
    peer_specs: Mapping[str, PeerSpec] | None = None,
    replica_snapshot: ReplicaSnapshot | None = None,
    dispatch_counter: DispatchCounter | None = None,
    mesh_snapshot: "RoutingSnapshot | None" = None,
) -> GatewayResponse:
    """Resolve the model to its ONE owning backend and try it exactly once.

    There is **no cross-backend failover** (issue #91):
    :func:`lobes.gateway._routing.order_backends` returns at most one backend —
    the owner of the resolved model — so this attempts that owner and nothing
    else. ``open_upstream`` is injected so this is unit-testable without sockets.

    The owner's verdict is authoritative, and the status mapping reflects that a
    request naming a model has exactly one honest place to go:

    * **unknown model id** → a **404** ``model_not_found`` generated HERE, before
      any routing (honesty h23). A non-empty ``model`` that is neither an alias
      nor any WIRED backend's served name (:func:`is_unknown_model`) was never
      advertised, so it must not be silently served under the default backend's
      weights. This is the deliberate converse of "advertised implies reachable"
      (issue #92): a model *listed in ``/v1/models``* never 404s, but one never
      listed *should*. Unknown-ness is decided against the ROUTING TABLE, never
      the readiness-filtered ``/v1/models`` list — a wired-but-dead backend
      (dropped from ``/v1/models`` but still in the table) is KNOWN and takes the
      retryable-503 path below, NOT this 404 (that is what keeps issue #91 fixed).
      A missing/blank ``model`` is *unspecified*, not unknown → it routes to
      ``default_model`` and is served.
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
    to be served instead of shed. Since t8 (#199) the shed has ONE exception: a
    role with a declared replica pool is FORWARDED to a selectable peer replica
    instead (:func:`_pooled_busy_dispatch`, spec c7/h6) — the 429 is reserved
    for "no replica anywhere is free", and 503 ``backend_unavailable`` for "no
    replica anywhere is up". A role with no pool sheds exactly as before.
    On the served path the ``X-Lobes-Tier`` /
    ``X-Lobes-Tier-Reason`` headers still travel with the response (prepended,
    streaming-safe). A plain model id, or ``pressure=None``, takes the existing
    non-tier path unchanged.

    Force-strict-tools (``cfg.force_strict_tools``, opt-in, colleague#320): once
    the owner is resolved, a ``/v1/chat/completions`` request whose owner is the
    ``primary`` (cortex) backend and whose body carries a non-empty ``tools``
    array is passed through :func:`inject_strict_tools`. When that ACTUALLY
    modifies the body (at least one ``tools[i].function`` lacked ``strict``),
    dialing is handed to :func:`_try_backend_with_strict_retry` instead of
    :func:`_try_backends` — it dials with the injected body, and on a 4xx/5xx
    matching a compile-failure signature retries once with the ORIGINAL
    un-injected body, relaying whichever response resulted. Every other
    request (knob off, non-primary lane, no tools, or every tool already
    carrying its own ``strict``) is entirely unaffected — same
    :func:`_try_backends` call as before this knob existed.

    The proxy data plane (``peer_specs``, proxy-lobes t6, issues #115/#127):
    when the requested model/alias resolves to a backend name in
    ``table.peer_proxied`` AND a :class:`PeerSpec` is wired for it (see
    :func:`peer_specs_from_table`), the request is FORWARDED to the declared
    peer via :func:`_proxy_to_peer` instead of taking the referral 404.
    Precedence is surgical: the unknown-id 404 (h23) still outranks proxying —
    :func:`_proxied_owner` checks it — while the proxy branch runs BEFORE the
    tier branch's pressure shedding, so a proxied request bypasses the LOCAL
    pressure policy entirely (pressure describes THIS box's load; the model
    runs on the peer, whose own gateway applies its own policy — shedding here
    too would double-gate the role on the wrong box's load). Non-proxied names
    — hosted, referral-only infeasible, unknown — take exactly the paths below,
    byte-identically; so does EVERY request when ``peer_specs`` is ``None``
    (every pre-t6 call site, and any deployment with no proxy config).

    The replica pool (``replica_snapshot``, cortex-replica-pool t7, issue
    #199; ``mesh_snapshot``, t13): AFTER the model has resolved to its single
    owning backend and BEFORE that backend is dialed, a role with either a
    declared ``<PREFIX>_PEER_ORIGINS`` set (a hand-built table only — t14
    deleted the env parsing that used to populate this in a real deployment)
    OR at least one mesh-verified plain-exposed member is placed by
    :func:`_pool_selection` — local, a declared peer replica, or a mesh
    replica of the SAME role, chosen from an O(1) cached snapshot. This is
    not a hole in #91: a peer enters the candidate set only when its
    live-probed fingerprint matches the local lane's, so a caller who asked
    for cortex is never answered by a different model — only by an identical
    one on another box. The alias and the raw served id take the identical
    path, because selection keys off the OWNING BACKEND NAME both resolve to
    (c31). An arriving ``X-Lobes-Proxied`` request skips selection entirely
    and is served locally (c4/h4). Pooled answers carry
    ``X-Lobes-Served-By`` (local) or ``X-Lobes-Proxied-By`` (forwarded), both
    with ``X-Lobes-Route-Reason``. With ``replica_snapshot`` ``None`` (every
    pre-pool call site), no ``*_PEER_ORIGINS`` declared, and no mesh
    (``mesh_snapshot`` ``None`` or nothing verified for the role), not one
    byte of any response changes — success or error path (h1).

    t8 gives that placement its failure semantics (:func:`_pool_dispatch`): a
    replica that fails PRE-DISPATCH (refused / timed out / 5xx before any
    bytes) is retried on the next selectable replica, at most once per replica
    and with the LOCAL replica an ordinary candidate; a replica that answered
    2xx and then dropped is NEVER replayed; a peer's own 4xx (its 429 shed
    included) rides back verbatim and is never retried, so a request produces
    at most ONE forward (c35/h27). ``X-Lobes-Route-Attempts`` appears when more
    than one replica was dispatched to, and exhausting them all is a 503
    ``backend_unavailable`` naming every attempt.
    """
    requested = extract_model(body)
    req_headers = list(req_headers)
    pooled = _peer_only_forward(
        table,
        cfg,
        peer_specs,
        path,
        req_headers,
        body,
        open_upstream,
        requested=requested,
        replica_snapshot=replica_snapshot,
        mesh_snapshot=mesh_snapshot,
        counter=dispatch_counter,
    )
    if pooled is not None:
        return pooled
    if peer_specs:
        proxied_name = _proxied_owner(table, peer_specs, requested)
        if proxied_name is not None:
            # A role that IS pooled but had nothing selectable reaches the
            # singular forward. Stamp the honest reason for that: without it a
            # trace cannot tell "the pool placed nothing because it is empty"
            # from "this deployment has no pool at all", two very different
            # operational states. REASON_NONE is the existing vocabulary's
            # word for "no replica was selected" — no new string. A role with
            # no pool declared is not in `pooled_backends`, so its response
            # stays byte-identical (h1/h5).
            fallthrough = (
                [(ROUTE_REASON_HEADER, REASON_NONE)]
                if proxied_name
                in pooled_backends(table, replica_snapshot, mesh_snapshot=mesh_snapshot)
                else []
            )
            return _proxy_to_peer(
                cfg,
                peer_specs[proxied_name],
                path,
                req_headers,
                body,
                open_upstream,
                extra_response_headers=fallthrough,
            )
    # --- mesh dispatch (W7 / W8) -------------------------------------------
    # When a RoutingSnapshot is present, the mesh can supply verified members
    # for roles that THIS BOX LACKS (W7) or augment the pool with peers for
    # roles THIS BOX HOSTS (W8).  Check the mesh BEFORE _resolve_served_or_early
    # so that the mesh intercepts the 404 that _feasibility_response produces
    # for infeasible roles.  This is a no-op when mesh_snapshot is None,
    # preserving the pre-mesh behaviour byte-for-byte.
    if mesh_snapshot is not None:
        from lobes.roles import BACKEND_ROLE, ROLE_BACKEND, ROLES

        # Suffixed-lane direct addressing (t8, issue #237): "{role}-{member}"
        # always resolves straight to that member's origin, independent of
        # whether the plain role name is currently placeable at all — a
        # caller that already knows which member it wants is never blocked
        # by a fingerprint disagreement among the others.
        if requested:
            lane = find_suffixed_lane(mesh_snapshot, requested, ROLES)
            if lane is not None:
                arriving = _arriving_hop_marker(req_headers)
                if arriving is not None:
                    target = _ForwardTarget(
                        name=lane.role, origin=lane.origin, served_name=requested
                    )
                    return GatewayResponse(
                        status=_PROXY_LOOP_STATUS,
                        headers=[("Content-Type", _CONTENT_TYPE_JSON)],
                        body=_proxy_loop_body(arriving, target),
                    )
                try:
                    mesh_cfg = _build_mesh_config()
                except MeshConfigError:
                    mesh_cfg = None
                join_key = (mesh_cfg and mesh_cfg.enabled and mesh_cfg.key) or ""
                # Finding 1 (review #252): `requested` is the GATEWAY-ONLY
                # suffixed alias ("cortex-thor") this box minted for direct
                # addressing — the destination member never declared that
                # name as one it serves, so rewriting the outbound body's
                # `model` to it (via `rewrite=True` below) sent the
                # destination a model id it does not recognise. Resolve to
                # the destination's canonical backend name instead — the
                # exact convention the plain (non-suffixed) mesh forward a
                # few lines below already uses (`served_name=owned_backend`)
                # — so both mesh-forward paths hand every destination a name
                # it actually serves. The suffixed name is kept only for THIS
                # box's own routing/response metadata (MESH_MEMBER_HEADER
                # below still names the member unambiguously).
                target = _ForwardTarget(
                    name=lane.role,
                    origin=lane.origin,
                    served_name=ROLE_BACKEND.get(lane.role, lane.role),
                    api_key=join_key,
                )
                return _proxy_to_peer(
                    cfg,
                    target,
                    path,
                    req_headers,
                    body,
                    open_upstream,
                    rewrite=True,
                    extra_response_headers=(
                        [(PROXIED_BY_HEADER, lane.origin)]
                        + mesh_markers(mesh_snapshot, lane.role, chosen_origin=lane.origin)
                        + [
                            (ROUTE_REASON_HEADER, "mesh-forwarded"),
                            (MESH_MEMBER_HEADER, lane.member),
                        ]
                    ),
                )

        # Resolve the model to its owning backend name using infeasible_owner
        # (which reuses resolve_model internally and handles role aliases like
        # "cortex" → backend name).  If this box lacks the backend (either
        # infeasible or unwired), check if mesh can forward.
        owned_backend = infeasible_owner(table, requested)
        if owned_backend is not None:
            role = BACKEND_ROLE.get(owned_backend, owned_backend)
            local_fp = _local_backend_fingerprint(replica_snapshot, owned_backend)
            placement = compute_role_placement(mesh_snapshot, role, local_fingerprint=local_fp)
            mesh_origins = placement.plain_origins
            if not mesh_origins and placement.suffixed:
                # Every verified member disagrees on this role's fingerprint
                # (or disagrees with this box's own local one) — the plain
                # role name is ambiguous and is refused rather than silently
                # picking one member (h1/h20/c46).
                return GatewayResponse(
                    status=404,
                    headers=[("Content-Type", _CONTENT_TYPE_JSON)],
                    body=_role_infeasible_body(
                        requested,
                        owned_backend,
                        suffixed_names=placement.suffixed_names(),
                    ),
                )
            if mesh_origins:
                # W7: this box lacks the role.  If mesh has verified members,
                # forward the request there instead of returning 404/502.
                # Single-hop guard: if the request already crossed one proxy,
                # refuse with 508 — same as _proxy_to_peer's loop check.
                arriving = _arriving_hop_marker(req_headers)
                if arriving is not None:
                    target = _ForwardTarget(
                        name=role,
                        origin=mesh_origins[0],
                        served_name=owned_backend,
                    )
                    return GatewayResponse(
                        status=_PROXY_LOOP_STATUS,
                        headers=[("Content-Type", _CONTENT_TYPE_JSON)],
                        body=_proxy_loop_body(arriving, target),
                    )
                # Select the first verified member.
                member_origin = mesh_origins[0]
                member_name = None
                for m in mesh_snapshot.members:
                    if m.origin == member_origin:
                        member_name = m.name
                        break
                member_name = member_name or member_origin

                try:
                    mesh_cfg = _build_mesh_config()
                except MeshConfigError:
                    mesh_cfg = None
                join_key = (mesh_cfg and mesh_cfg.enabled and mesh_cfg.key) or ""
                target = _ForwardTarget(
                    name=role,
                    origin=member_origin,
                    served_name=owned_backend,
                    api_key=join_key,
                )
                resp = _proxy_to_peer(
                    cfg,
                    target,
                    path,
                    req_headers,
                    body,
                    open_upstream,
                    rewrite=True,
                    extra_response_headers=(
                        [(PROXIED_BY_HEADER, member_origin)]
                        + mesh_markers(mesh_snapshot, role, chosen_origin=member_origin)
                        + [
                            (ROUTE_REASON_HEADER, "mesh-forwarded"),
                            (MESH_MEMBER_HEADER, member_name),
                        ]
                    ),
                )
                return resp
            # Boot window (t3): no ROUTABLE mesh member, but a member that
            # announced this role has not been probed yet — answer "not yet"
            # (503 role_unverified, naming it) instead of falling through to
            # the terminal "never" 404 below.
            pending = _role_unverified_response(
                mesh_snapshot, role, placement, requested, owned_backend
            )
            if pending is not None:
                return pending

    early, served, tier_headers, local_busy = _resolve_served_or_early(
        table,
        cfg,
        path,
        req_headers,
        body,
        open_upstream,
        requested=requested,
        pressure=pressure,
        override=override,
        replica_snapshot=replica_snapshot,
        mesh_snapshot=mesh_snapshot,
        dispatch_counter=dispatch_counter,
    )
    if early is not None:
        return early

    if mesh_snapshot is not None and served is not None:
        # Resolve the owning role for this backend name.  The mapping is
        # imported lazily because lobes.roles creates an import cycle with
        # this package's __init__.py — it only "worked" when something else
        # happened to import lobes.gateway first.
        from lobes.roles import BACKEND_ROLE

        role = BACKEND_ROLE.get(served, served)
        # Order backends to see if this box actually hosts the role.  An
        # empty list means the backend is either infeasible (declared off by
        # the per-machine profile) or completely unwired — either way, this
        # box cannot serve it locally.
        ordered_here = order_backends(table, served)
        if not ordered_here:
            # W7: this box lacks the role.  If mesh has verified members,
            # forward the request there instead of returning 404/502.
            local_fp = _local_backend_fingerprint(replica_snapshot, served)
            placement = compute_role_placement(mesh_snapshot, role, local_fingerprint=local_fp)
            mesh_origins = placement.plain_origins
            if not mesh_origins and placement.suffixed:
                return GatewayResponse(
                    status=404,
                    headers=[("Content-Type", _CONTENT_TYPE_JSON)],
                    body=_role_infeasible_body(
                        requested,
                        served,
                        suffixed_names=placement.suffixed_names(),
                    ),
                )
            if mesh_origins:
                # Single-hop guard: if the request already crossed one proxy,
                # refuse with 508 — same as _proxy_to_peer's loop check.
                arriving = _arriving_hop_marker(req_headers)
                if arriving is not None:
                    target = _ForwardTarget(
                        name=role,
                        origin=mesh_origins[0],
                        served_name=served,
                    )
                    return GatewayResponse(
                        status=_PROXY_LOOP_STATUS,
                        headers=[("Content-Type", _CONTENT_TYPE_JSON)],
                        body=_proxy_loop_body(arriving, target),
                    )
                # Select the first verified member (already verified by the
                # probe; no further capability check needed here).
                member_origin = mesh_origins[0]
                member_name = None
                for m in mesh_snapshot.members:
                    if m.origin == member_origin:
                        member_name = m.name
                        break
                member_name = member_name or member_origin

                try:
                    mesh_cfg = _build_mesh_config()
                except MeshConfigError:
                    mesh_cfg = None
                join_key = (mesh_cfg and mesh_cfg.enabled and mesh_cfg.key) or ""
                target = _ForwardTarget(
                    name=role,
                    origin=member_origin,
                    served_name=served,
                    api_key=join_key,
                )
                resp = _proxy_to_peer(
                    cfg,
                    target,
                    path,
                    req_headers,
                    body,
                    open_upstream,
                    rewrite=True,
                    extra_response_headers=(
                        [(PROXIED_BY_HEADER, member_origin)]
                        + mesh_markers(mesh_snapshot, role, chosen_origin=member_origin)
                        + [
                            (ROUTE_REASON_HEADER, "mesh-forwarded"),
                            (MESH_MEMBER_HEADER, member_name),
                        ]
                    ),
                )
                return resp
            # Boot window (t3): same "not yet" answer as the alias path above,
            # for a request that named the raw checkpoint id this member
            # announced.
            pending = _role_unverified_response(mesh_snapshot, role, placement, requested, served)
            if pending is not None:
                return pending
            # No verified mesh member — this box lacks the role and no peer
            # can serve it either.  Let the normal flow produce the 404.
        else:
            # W8: this box hosts the role and mesh has verified members.
            # Merge local and mesh candidates so select_replica can rank
            # across the whole pool.  The existing _pool_dispatch path then
            # chooses local or peer as normal.

            def _make_mesh_candidates(
                snap: "RoutingSnapshot | None",
                role_name: str,
                *,
                plain_origins: "tuple[str, ...]" = (),
            ) -> list[ReplicaState]:
                """Build ReplicaState objects for members placeable PLAIN.

                Only origins :func:`~lobes.gateway._mesh_routing.compute_role_placement`
                put in the plain pool are merged here — a member whose
                fingerprint disagrees is exposed only under its suffixed
                name (t8, #237) and must never silently join the ranked
                plain-pool candidate set.
                """
                out: list[ReplicaState] = []
                if snap is None:
                    return out
                for m in snap.members:
                    if role_name in m.verified_roles and m.origin in plain_origins:
                        out.append(
                            ReplicaState(
                                origin=m.origin,
                                local=False,
                                ready=True,
                                busy=False,
                                health="ok",
                                running=0,
                                waiting=0,
                                fingerprint=None,
                                compatible=True,
                                reason="",
                                last_seen=0.0,
                                weight=8.0,
                                calibrated=True,
                            )
                        )
                return out

            local_fp_for_merge = _local_backend_fingerprint(replica_snapshot, served)
            merge_placement = compute_role_placement(
                mesh_snapshot, role, local_fingerprint=local_fp_for_merge
            )
            mesh_cands = _make_mesh_candidates(
                mesh_snapshot, role, plain_origins=merge_placement.plain_origins
            )
            if mesh_cands:

                def _merged_snapshot(backend_name: str):
                    local = replica_snapshot(backend_name) if replica_snapshot else ()
                    if backend_name == served:
                        return tuple(local) + tuple(mesh_cands)
                    return local

                replica_snapshot = _merged_snapshot

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
            body=_error_body("no backend owns the requested model", []),
            attempts=[],
        )

    streaming = is_streaming(body)
    fwd_body = rewrite_model(body, served)
    fwd_headers = filter_headers(req_headers)

    def dial_local(headers: list[tuple[str, str]]):
        """Dial THIS box's own replica. Passed into the pool loop so the local
        replica is an ordinary retry candidate, and reused verbatim below for
        the unpooled / nothing-selectable paths — one dialer, one behaviour.

        ``headers`` carries the pool markers for THIS attempt; prepending them
        to the upstream's own headers puts them on every local outcome (relay,
        owner-down 503) and keeps them streaming-safe (nothing is emitted after
        the first chunk)."""
        return _dial_owner(
            ordered, cfg, path, fwd_body, fwd_headers, open_upstream, streaming, headers
        )

    # --- replica pool (t7/t8, #199): local-vs-peer placement, before dialing ---
    outcome = _pool_dispatch(
        _RequestCtx(table, cfg, path, req_headers, body, open_upstream),
        backend_name=ordered[0].name,
        served=served,
        tier_headers=tier_headers,
        replica_snapshot=replica_snapshot,
        mesh_snapshot=mesh_snapshot,
        local_busy=local_busy,
        dial_local=dial_local,
        counter=dispatch_counter,
    )
    if isinstance(outcome, GatewayResponse):
        return outcome
    release = _no_release
    if isinstance(outcome, _PoolFallthrough):
        # Pooled, but nothing was dispatched: dial the local owner exactly as
        # the pre-pool release would, honestly marked `none` rather than
        # claiming a selection happened (t7's behaviour, kept).
        tier_headers = _stamp_pool_headers(table, outcome.reason) + tier_headers
        # ...and COUNT it. No placement is invented here — the markers above
        # still say what actually happened — but the request is about to run
        # on this box's own engine, and the in-flight tally is a record of
        # what the engine is executing, not of what the router decided. Both
        # fallthroughs reach this line and both are genuine local work: a
        # single-hop marked arrival forwarded BY a peer (which would otherwise
        # be invisible to this box's own snapshot until the next probe, so
        # this box kept selecting an already-loaded local replica), and a
        # fleet with no room anywhere, which `d5` queues locally rather than
        # shedding. An UNPOOLED request (outcome None) is untouched — h1.
        release = (dispatch_counter or _uncounted)(ordered[0].name, ordered[0].base_url)

    response, attempts = _counted_local_dial(dial_local, tier_headers, release)
    if response is not None:
        return response

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


def handle_audio_request(
    table: RoutingTable,
    cfg: ServerConfig,
    peer_specs: Mapping[str, PeerSpec] | None,
    path: str,
    req_headers: Iterable[tuple[str, str]],
    body: bytes,
    open_upstream: OpenUpstream,
    *,
    audio_ready_probe: Callable[[], bool | None] | None = None,
    mesh_snapshot: "RoutingSnapshot | None" = None,
) -> GatewayResponse:
    """Route one ``/v1/audio/*`` POST — per-ROLE since issue #129.

    ``/v1/audio/speech`` is the tts lane and ``/v1/audio/transcriptions`` the
    stt lane (:func:`lobes.gateway._routing.audio_role_for_path`), and the two
    move between boxes independently — the live trigger was Chatterbox (tts)
    on a peer while Parakeet (stt) stays local, which the one namespace-wide
    ``AUDIO_URL`` cannot express. Precedence mirrors the model-routed lanes:

    * **proxied** (the role is in ``table.peer_proxied`` with a built spec) —
      forward via :func:`_proxy_to_peer` with the body VERBATIM
      (``rewrite=False``): credential swap, single-hop guard, and
      ``X-Lobes-Proxied-By`` attribution all apply, so the four AUDIO_URL
      contract violations recorded on #129 are impossible on this lane;
    * **declared off** (``STT_/TTS_FEASIBLE=false`` → ``table.infeasible``)
      and not proxied — the honest 404 ``role_infeasible`` with ``hosted_by``
      when a peer origin is declared, never half-served;
    * **otherwise** — the legacy local ``AUDIO_URL`` route
      (:func:`handle_audio_post`), byte-identical to pre-#129 behaviour;
      ``audio_ready_probe`` is called only on this branch (a proxied or
      refused lane never pays for a local readiness probe).
    """
    role = audio_role_for_path(path)
    spec = (peer_specs or {}).get(role) if role else None
    if role is not None and role in table.peer_proxied and spec is not None:
        return _proxy_to_peer(
            cfg,
            spec,
            path,
            req_headers,
            body,
            open_upstream,
            rewrite=False,  # path-routed lane: multipart/TTS JSON verbatim
        )
    if role is not None and role in table.infeasible:
        peer_origin = table.peer_origins.get(role) or _mesh_referral_origin(mesh_snapshot, role)
        return GatewayResponse(
            status=404,
            headers=[("Content-Type", _CONTENT_TYPE_JSON)],
            body=_role_infeasible_body(role, role, peer_origin),
        )
    audio_ready = audio_ready_probe() if audio_ready_probe is not None else None
    return handle_audio_post(cfg, path, req_headers, body, open_upstream, audio_ready=audio_ready)


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
    if RENDER_TASK in tasks:
        # The render facade (issue #82, t9) — advertised only when the tenant is
        # actually wired, like every other task family here. Job-scoped by
        # construction: there is no /history or /view spelling to advertise.
        eps += [
            f"POST {RENDER_PATH}",
            f"POST {RENDER_PATH}/uploads/image",
            f"POST {RENDER_PATH}/jobs/{{job_id}}/cancel",
            f"GET {RENDER_PATH}/jobs/{{job_id}}",
            f"GET {RENDER_PATH}/jobs/{{job_id}}/artifacts",
            f"GET {RENDER_PATH}/jobs/{{job_id}}/artifacts/{{filename}}",
        ]
    # Per-role audio honesty (issue #129): each lane is advertised iff it is
    # answerable HERE — served by the local overlay (and not declared off) or
    # forwarded to a declared peer. A declared-off, unproxied lane 404s
    # role_infeasible and must not be advertised.
    if (audio and "stt" not in table.infeasible) or "stt" in table.peer_proxied:
        eps.append("POST /v1/audio/transcriptions")
    if (audio and "tts" not in table.infeasible) or "tts" in table.peer_proxied:
        eps.append("POST /v1/audio/speech")
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

    Capacity (capacity-relative pool routing, t5): each HOSTED backend row
    additionally carries this box's own declared max-active-requests capacity
    under :data:`lobes.gateway._replicas.PEER_CAPACITY_KEY`. That is the whole
    discovery path a peer needs — ``ReplicaCache`` already probes ``/status``
    every refresh, so no new endpoint and no new probe exist for a peer to
    dial. The key is strictly ADDITIVE: a box that declares no capacity
    publishes none, an older lobes ignores what it does not know, and
    ``_replicas.py`` reads an absent capacity as uncalibrated rather than as
    zero. See :func:`_published_capacity` for what is deliberately withheld.
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
    busy_partial = False
    for b, st in zip(members, results):
        metrics = st.get("metrics") or {}
        # A non-vLLM engine may not export in-flight counts at all (see
        # lobes._metrics). Such a backend must not silently contribute 0 to the
        # fleet aggregate — the total is flagged partial so /status never implies
        # a lane is idle when it is simply unmeasured.
        unknown = _metrics.unsupported_fields(metrics)
        if "running" in unknown or "waiting" in unknown:
            busy_partial = True
        running += int(metrics.get("running", 0) or 0)
        waiting += int(metrics.get("waiting", 0) or 0)
        row = {
            "name": b.name,
            "task": b.task,
            "served_name": b.served_name,
            "health": st.get("health", "unreachable"),
            "metrics": st.get("metrics"),
        }
        row.update(_published_capacity(table, cfg, b.name))
        backends.append(row)
    busy: dict = {"running": running, "waiting": waiting}
    if busy_partial:  # added only when true → an all-vLLM fleet's payload is unchanged
        busy["partial"] = True
    payload = {
        "object": "lobes.fleet_status",
        "default_model": table.default_model,
        "busy": busy,
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


def _published_capacity(table: RoutingTable, cfg: ServerConfig, name: str) -> dict[str, float]:
    """This box's declared capacity for backend *name*, as a ``/status`` fragment.

    An empty dict — i.e. no key at all — in three cases, each deliberate:

    * **nothing declared.** ``<PREFIX>_MAX_ACTIVE`` is unset, so there is no
      measured number to publish. Fabricating one (``1.0``, say) would arrive
      at a peer as a CALIBRATED one-slot capacity and starve this box; an
      absent key arrives as "uncalibrated", which is the truth.
    * **this box does not host the role.** A lane declared infeasible has no
      local replica at all, so advertising room on it would invite a peer to
      forward work nothing here can serve.
    * **a never-proxied backend.** :data:`~lobes.gateway._config.NEVER_PROXIED_BACKENDS`
      names the roles that are absent from every cross-box channel by
      decision rather than by accident; capacity is one more such channel, so
      it consults the same constant instead of re-deriving the rule. (It is
      empty today — the 2026-08-20 ``d1`` reversal made ``hand`` proxyable —
      so this arm is a guard for the next role that opts out, not a live
      exclusion.)
    """
    if name in table.infeasible or name in NEVER_PROXIED_BACKENDS:
        return {}
    declared = cfg.local_capacities.get(name)
    return {PEER_CAPACITY_KEY: float(declared)} if declared is not None else {}


# --- role capabilities (the #81 role→endpoint contract) --------------------

# GET /capabilities reuses lobes.roles.build_role_registry — the SAME builder
# the CLI's `lobes capabilities --json` calls — so the two payloads are
# exactly the same shape: a dict keyed by role, each value the full RoleInfo
# field set. The route derives a client-reachable origin (#87) and a live audio
# readiness signal (#89) and hands them to the builder; the pure function keeps
# its config-derived defaults so the CLI/unit path is unchanged.


# A legitimate HTTP ``Host`` header is a bare authority: a DNS hostname or IPv4
# literal (dot-separated alphanumeric/hyphen labels — RFC 1123, which an IPv4
# literal's digit-only labels already satisfy) or a bracketed IPv6 literal,
# each optionally followed by ``:<port>`` (1-5 digits). Nothing in that grammar
# permits ``/``, ``@``, whitespace, control characters, ``<``/``>``, ``?``,
# ``#``, or backslashes, so a single allowlist regex both recognises a
# well-formed host AND excludes every character class a path-traversal,
# userinfo-credential-injection, header-injection (CRLF), XSS, or
# query-string payload needs. See :func:`reachable_origin` for why this
# exists (SonarCloud S5131).
_HOST_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
_HOSTNAME = rf"{_HOST_LABEL}(?:\.{_HOST_LABEL})*"
_IPV6_LITERAL = r"\[[0-9A-Fa-f:]+\]"
_PORT = r"(?::[0-9]{1,5})?"
_VALID_HOST_HEADER_RE = re.compile(rf"(?:{_HOSTNAME}|{_IPV6_LITERAL}){_PORT}")


def _is_valid_host_header(host: str) -> bool:
    """True when ``host`` is a well-formed ``hostname[:port]`` authority.

    Used to gate :func:`reachable_origin`'s Host-header echo — see there for
    the reflection risk this guards against.
    """
    return _VALID_HOST_HEADER_RE.fullmatch(host) is not None


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

    ``public_url`` is trusted operator config (set via the deployment's own
    ``.env``, never attacker-reachable) so it is never validated and always
    wins first, unchanged — that precedence is #92 target c29/h25 and is
    covered by ``test_reachable_origin_public_url_wins_over_host`` /
    ``test_capabilities_public_url_wins_over_host_end_to_end``.

    ``host_header``, by contrast, is fully attacker-controlled: any client can
    set an arbitrary ``Host:`` value, and this function's return value is
    reflected verbatim into every role's ``endpoint`` in the JSON response
    (:func:`capabilities_payload`). Echoing it unsanitised is exactly
    SonarCloud rule ``pythonsecurity:S5131`` ("Change this code to not reflect
    unsanitized user-controlled data") — a scraping client could be handed an
    attacker's origin to dial, or a payload (path traversal, script markup, a
    userinfo-style credential-injection host like
    ``127.0.0.1:8001@attacker.test``) smuggled through an otherwise-trusted
    contract. The remediation is the standard S5131 fix: constrain the tainted
    value to a strict allowlist (:func:`_is_valid_host_header`, a bare
    ``hostname[:port]``/``[ipv6][:port]`` authority) before it can reach the
    response. A ``Host`` header that fails validation is treated exactly like
    a missing one — it falls through to ``None``, and the caller advertises an
    empty endpoint (never a fabricated or attacker-supplied one) rather than
    guessing at a "sanitised" rewrite of untrusted input.
    """
    if public_url:
        return public_url.rstrip("/")
    if host_header and _is_valid_host_header(host_header):
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


def _local_backend_fingerprint(
    replica_snapshot: "ReplicaSnapshot | None",
    backend_name: str,
) -> "object | None":
    """This box's OWN fingerprint for *backend_name*, or ``None`` if unhosted.

    ``replica_snapshot`` here is the ``handle_post``-shaped callable (backend
    name -> that backend's replica tuple), NOT the role-keyed mapping
    :func:`_pooled_peer_advert` reads — the two channels use the same
    underlying data with different keys, so this looks up by BACKEND. The
    local entry (``.local is True``) is this box's own served fingerprint —
    the reference :func:`~lobes.gateway._mesh_routing.compute_role_placement`
    uses to decide which mesh members agree with what THIS box actually
    serves. ``None`` when there is no snapshot, no entry for the backend, or
    no local replica in it (this box does not host it at all).
    """
    if replica_snapshot is None:
        return None
    try:
        states = replica_snapshot(backend_name)
    except Exception:  # nosec B110 — best-effort: a broken snapshot never blocks placement
        return None
    for state in states or ():
        if getattr(state, "local", False):
            return getattr(state, "fingerprint", None)
    return None


def _pooled_peer_advert(
    table: RoutingTable,
    replica_snapshot: "Mapping[str, tuple[ReplicaState, ...]] | None",
) -> tuple[dict[str, bool | None], dict[str, int | None]]:
    """``(ready, context)`` per BACKEND name for every pooled role, or two empty dicts.

    Keyed by backend name because that is the vocabulary
    :func:`lobes.roles._role_signals` reads, while ``replica_snapshot`` is
    keyed by ROLE — the two are translated here rather than at either end.

    Only NON-LOCAL replicas are folded: a box that hosts the role has its own
    lane readiness already, and the peer channel exists precisely for the
    boxes that do not. A replica that is not ``compatible`` contributes
    nothing to either value — pooling an unknown is what #199 h11 forbids, and
    advertising a window this box would never route to is the #220 lie in a
    new costume.
    """
    if not replica_snapshot or not table.replica_origins:
        return {}, {}
    # Deferred import — see the module-level NOTE on the lobes.roles cycle.
    from lobes.roles import BACKEND_ROLE

    ready: dict[str, bool | None] = {}
    context: dict[str, int | None] = {}
    for backend in table.replica_origins:
        states = replica_snapshot.get(BACKEND_ROLE.get(backend, backend)) or ()
        usable = [s for s in states if not s.local and s.compatible]
        if not usable:
            continue
        ready[backend] = any(s.ready for s in usable)
        windows = {
            s.fingerprint.max_model_len
            for s in usable
            if s.fingerprint is not None and s.fingerprint.max_model_len
        }
        context[backend] = windows.pop() if len(windows) == 1 else None
    return ready, context


def capabilities_payload(
    table: RoutingTable,
    cfg: ServerConfig,
    env: Mapping[str, str] | None = None,
    *,
    gateway_url: str | None = None,
    audio_ready: bool | None = None,
    backend_ready: Mapping[str, bool | None] | None = None,
    peer_context: Mapping[str, int | None] | None = None,
    replica_snapshot: Mapping[str, tuple[ReplicaState, ...]] | None = None,
    mesh_snapshot: "RoutingSnapshot | None" = None,
) -> dict:
    """The nine first-class roles (issue #81), resolved via the shared registry.

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
    the live readiness snapshot for the five gateway-fronted roles (issue #92),
    keyed by internal ``Backend`` name — exactly what
    :meth:`lobes.gateway._readiness.ReadinessCache.current` returns, so the HTTP
    route passes it straight through; when ``None`` each role's ``ready`` falls
    back to ``loaded`` (the CLI/unit path). All three signal kwargs default to
    ``None`` so this pure function's shape is unchanged for its non-HTTP callers.

    Proxied roles (t6, issues #115/#127): for each name in
    ``table.peer_proxied`` the ``backend_ready`` snapshot's value IS the live
    PEER probe verdict — :meth:`ReadinessCache.current` merges the peer store
    over the local one for exactly those names (the peer thread's
    :func:`~lobes.gateway._readiness.probe_peer_ready` result) — so this
    function slices those entries into the builder's separate ``peer_ready``
    channel, and a proxied role's ``ready`` honestly reflects the live
    proxied-path probe (h2). With ``backend_ready`` omitted, or with no
    proxied names, nothing is derived and every payload is unchanged.

    ``peer_context`` (issue #220) is the other half of the same advert —
    :meth:`ReadinessCache.current_peer_context`, keyed by the same internal
    backend names — and is passed straight to the builder, which applies it
    ONLY to a role in ``table.peer_proxied``. It exists because a role this box
    does not host has no local ``<PREFIX>_MAX_MODEL_LEN`` to read, so the local
    computation falls back to the CATALOG's native ceiling and advertises a
    window the peer never serves. ``None`` (its default, and every deployment
    with no proxied roles) leaves every payload byte-identical.
    """
    # deferred imports — see the module-level NOTE
    from lobes.roles import (
        ROLES,
        annotate_mesh_naming,
        annotate_peer_referrals,
        annotate_replicas,
        build_role_registry,
        role_payload,
    )

    resolved_env = os.environ if env is None else env
    peer_ready = None
    if backend_ready is not None and table.peer_proxied:
        peer_ready = {name: backend_ready.get(name) for name in table.peer_proxied}
    # The POOLED advert (peer-only-replica-pools): a role served by N replicas
    # has no single peer whose /capabilities can be relayed, so the honest
    # answer is folded from the snapshot the pool already probes — ready is
    # true when ANY compatible replica is ready, and context is the
    # fingerprint-AGREED max_model_len (agreed by construction: max_model_len
    # is a disqualifying field, so two compatible replicas cannot disagree on
    # it). This channel WINS over the singular peer relay for a pooled role:
    # #220's per-peer probe reads one declared origin, which is not the
    # authority once several serve the same role. With no pool it is empty and
    # every payload keeps its pre-pool bytes.
    pooled_ready, pooled_context = _pooled_peer_advert(table, replica_snapshot)
    if pooled_ready:
        peer_ready = {**(peer_ready or {}), **pooled_ready}
    if pooled_context:
        peer_context = {**(peer_context or {}), **pooled_context}
    registry = build_role_registry(
        table,
        cfg,
        env=resolved_env,
        gateway_url=gateway_url,
        audio_ready=audio_ready,
        backend_ready=backend_ready,
        peer_ready=peer_ready,
        peer_context=peer_context,
    )
    payload = {role: role_payload(registry[role]) for role in ROLES}
    # Opt-in honest referral (mesh-brain t3): annotate each unhosted
    # (feasible=false) role with the OPERATOR-DECLARED peer origin that hosts
    # it (table.peer_origins). With no peer config (the default) this is a
    # no-op and the payload stays byte-identical to the pre-referral contract.
    payload = annotate_peer_referrals(payload, table)
    # The replica pool (t8, issues #199 c9/c33): the additive per-role
    # `fingerprint` + `replicas` keys, from the LIVE background snapshot when
    # this process has one. `fingerprint` is what makes the pool checkable
    # cross-box — a peer reads it off THIS box's /capabilities to decide
    # whether our replica is compatible with its own (c33/h25), which is why
    # it is published even for a role whose own peer list is empty. With no
    # pool declared and no snapshot, `annotate_replicas` is a no-op for every
    # role and the payload stays byte-identical to the pre-pool contract (h1).
    # Finding 10 (review #252): tell annotate_replicas whether the mesh is
    # enabled at all so an ordinary hosted role — no declared replica pool,
    # the common single-box case — still publishes a fingerprint for a
    # peer's verification probe to compare against.
    payload = annotate_replicas(
        payload, table, replica_snapshot, mesh_enabled=mesh_snapshot is not None
    )
    # Suffixed-lane naming (t8, issue #237): the additive per-role
    # `member`/`suffixed_lanes` keys, from the mesh routing snapshot when this
    # process has one. With mesh disabled (`mesh_snapshot is None`, every
    # pre-t8 deployment) this is a no-op and the payload stays byte-identical.
    local_fingerprints = {
        role: next((s for s in (replica_snapshot or {}).get(role, ()) if s.local), None)
        for role in ROLES
    }
    return annotate_mesh_naming(
        payload, as_routing_snapshot(mesh_snapshot), local_fingerprints=local_fingerprints
    )


# --- the unmatched-route 404 body (SonarCloud S5131, companion to
# reachable_origin/_is_valid_host_header above) -----------------------------

# do_GET's fallback branch names the route it couldn't match — useful for a
# caller who mistyped a path or hit a route this deployment doesn't serve.
# But ``route`` comes straight off the request line (``self.path``, split
# before the query string) with NO decoding or validation, so echoing it
# verbatim reflects fully attacker-controlled bytes back into the response —
# the same taint shape SonarCloud rule ``pythonsecurity:S5131`` flagged for
# the Host header above, and the same remediation applies: constrain the
# tainted value to a strict allowlist before it can reach the response,
# rather than trying to escape or deny individual dangerous characters. A
# legitimate unmatched route is always a short, plain path
# (letters/digits/``/``/``-``/``_``/``.``) — nothing in that shape needs
# ``<``/``>``/quotes/backslashes/whitespace/control characters, so a route
# outside this allowlist, or one long enough to be a flood/log-noise attempt
# rather than a typo, was never one a genuine caller needed named back to
# it. ``_MAX_ECHOED_ROUTE_LEN`` is a defensive cap only — HTTP sets no limit
# here — chosen well above any real route this server declares.
_MAX_ECHOED_ROUTE_LEN = 200
_VALID_ROUTE_ECHO_RE = re.compile(r"/[A-Za-z0-9/_.-]*")


def _not_found_body(route: str) -> dict:
    """The ``{"error": {...}}`` 404 payload for a GET route that matched none
    of the handled endpoints.

    Echoes ``route`` only when it passes the allowlist above; otherwise the
    message falls back to a route-free ``"not found"`` — exactly as if the
    caller had sent no path detail at all, never a "sanitised" rewrite of
    attacker-supplied input. Either way the contract shape (``error.message``
    + ``error.type == "not_found"``) is unchanged, so callers that only
    branch on ``type`` (never on the message text) are unaffected.
    """
    safe_route = (
        route
        if len(route) <= _MAX_ECHOED_ROUTE_LEN and _VALID_ROUTE_ECHO_RE.fullmatch(route)
        else None
    )
    message = f"not found: {safe_route}" if safe_route is not None else "not found"
    return {"error": {"message": message, "type": "not_found"}}


# --- the /v1/render facade (issue #82, t9) ---------------------------------
#
# The gateway fronts a ComfyUI that has NO authentication of any kind. The
# fleet bearer answers WHO may reach the lane; it says nothing about WHAT a
# caller may read once inside. ComfyUI's own surfaces make that gap total:
# ``GET /history`` lists every prompt the box has ever run (other agents'
# included) and ``GET /view`` serves every output by a shared, counter-named
# filename (``flux_output_00001_.png``, ``_00002_``, …). A transparent prefix
# relay would therefore hand any bearer-holder the entire render history of
# the machine (spec claim c35).
#
# So this facade is JOB-SCOPED (c38). Three properties carry that, and each is
# covered by a negative control in tests/test_gateway_render_facade.py:
#
# 1. the gateway MINTS the outward job id and keeps ComfyUI's ``prompt_id``
#    to itself, so the upstream id space is not addressable at all;
# 2. status / artifact-index / artifact-bytes / cancel are served only for ids
#    :class:`RenderJobRegistry` recorded as ISSUED — an id this process did
#    not mint is refused without dialing ComfyUI;
# 3. an artifact name is checked against THAT JOB's own outputs, read back
#    from the upstream, before ``/view`` is dialed — so a valid job id still
#    cannot walk to the next counter's file.
#
# There is no outward spelling of ``/history``, ``/queue`` or a caller-
# parameterised ``/view`` (see :func:`parse_render_route`, an allowlist).

# ComfyUI's native endpoints, dialed only from here. The 0.33.2 ``/api/jobs``
# surface is what innereye itself prefers; the older ``/history`` + ``/queue``
# fallback is deliberately NOT wired outward, and `extract_job_artifacts`
# parses either shape should a deployment's upstream answer in the old one.
_COMFY_SUBMIT_PATH = "/prompt"
_COMFY_JOB_PATH = "/api/jobs/{}"
_COMFY_CANCEL_PATH = "/api/jobs/{}/cancel"
_COMFY_VIEW_PATH = "/view"
_COMFY_UPLOAD_PATH = "/upload/image"

# The render lane's READINESS probe path (task t11, issue #92 c9/h20). The
# gateway's shared ``ReadinessCache`` default probes ``/health`` on every
# backend, but ComfyUI serves no such route — MEASURED live against ComfyUI
# 0.33.2: ``/health`` 404s, while ``/``, ``/system_stats`` and
# ``/object_info`` all answer 200. ``/object_info`` is picked over the other
# two survivors deliberately: it lists every importable node and is only
# complete once ComfyUI's custom-node import pass finishes, so a 200 there is
# a genuine READINESS signal ("the server can actually run a workflow"), not
# merely a LIVENESS one ("the process accepted a socket") the way ``/`` would
# be. See ``ReadinessCache``'s ``paths`` constructor argument, below, for the
# wiring — this is the ONLY backend that gets a non-default path.
_COMFY_READY_PATH = "/object_info"

# The gateway backend NAME of the render tenant (its Colleague role name too).
_RENDER_BACKEND = "innereye"

# Seconds a caller should wait before retrying a render request that could not
# reach the backend. Mirrors BACKEND_UNAVAILABLE_RETRY_AFTER_SECONDS.
RENDER_RETRY_AFTER_SECONDS: int = 5

# How many issued job ids one gateway process remembers. Past this the oldest
# is evicted and its id stops resolving — the facade FAILS CLOSED (refuses),
# never open.
_MAX_ISSUED_RENDER_JOBS = 4096


class RenderJobRegistry:
    """The ComfyUI ``prompt_id`` behind each job id THIS gateway issued.

    Thread-safe by construction: the gateway is a
    :class:`~http.server.ThreadingHTTPServer`, so a submit and a status poll
    genuinely race. Every read and write takes the same lock, and the
    ``OrderedDict`` is never exposed.

    **Restart behaviour, stated plainly (honesty h28/h31).** This mapping lives
    in process memory ONLY. A gateway restart — a ``lobes up gateway``, a crash,
    a container recreate — forgets every id it ever issued, so a job id handed
    out before the restart is refused ``render_job_not_found`` afterwards even
    though ComfyUI still holds that job and its artifacts. Nothing is lost on
    disk: the outputs remain in the operator's bind-mounted output tree, which
    is the exposure the deployment intends for them. This is a deliberate
    fail-CLOSED choice over persisting the map: the alternative — a store the
    gateway does not otherwise have — would have to survive exactly the events
    that make the gateway's own state suspect, and a mis-restored map would
    hand one caller another's job. The same fail-closed rule covers eviction
    past :data:`_MAX_ISSUED_RENDER_JOBS`.
    """

    def __init__(self, limit: int = _MAX_ISSUED_RENDER_JOBS) -> None:
        self._lock = threading.Lock()
        self._issued: "OrderedDict[str, str]" = OrderedDict()
        self._limit = max(1, limit)

    def issue(self, prompt_id: str) -> str:
        """Mint an opaque job id for ``prompt_id`` and remember the pairing."""
        job_id = uuid.uuid4().hex
        with self._lock:
            self._issued[job_id] = prompt_id
            while len(self._issued) > self._limit:
                self._issued.popitem(last=False)
        return job_id

    def prompt_id(self, job_id: str) -> str | None:
        """The upstream id for ``job_id``, or ``None`` if this process never
        issued it (an unknown, a forgotten, or another box's id — all three are
        the same answer here, which is the point)."""
        with self._lock:
            return self._issued.get(job_id)


def _render_error_body(message: str, error_type: str) -> bytes:
    """OpenAI-shaped error body for the render facade.

    Note what the ``render_job_not_found`` message deliberately does NOT say:
    whether the id was never issued, was evicted, or belongs to someone else.
    Distinguishing them would make the refusal an existence oracle over the
    box's render history — the very thing the job scoping exists to close.
    """
    return json.dumps(
        {"error": {"message": message, "type": error_type, "code": error_type}}
    ).encode()


def _render_json(status: int, payload: dict) -> GatewayResponse:
    return GatewayResponse(
        status=status,
        headers=[("Content-Type", _CONTENT_TYPE_JSON)],
        body=json.dumps(payload).encode(),
    )


def _render_error(status: int, error_type: str, message: str) -> GatewayResponse:
    return GatewayResponse(
        status=status,
        headers=[("Content-Type", _CONTENT_TYPE_JSON)],
        body=_render_error_body(message, error_type),
    )


def _render_relay_headers(headers: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """Upstream headers safe to pass back on a relayed artifact.

    Drops ``Content-Length``/``Transfer-Encoding`` on top of the usual
    hop-by-hop filter: the streaming relay re-chunks the body and sets its own
    framing, so an upstream length header would contradict the wire.
    """
    return [
        (k, v)
        for k, v in filter_headers(headers)
        if k.lower() not in ("content-length", "transfer-encoding")
    ]


def _render_upstream_headers(body: bytes, content_type: str | None) -> list[tuple[str, str]]:
    """The headers the gateway sends INTO ComfyUI — built, never forwarded.

    The caller's own headers are deliberately not passed through. ComfyUI has
    no authentication, so relaying the fleet ``Authorization`` into it would
    put the key in a process that neither checks nor needs it; and the caller's
    ``Host``/``Content-Length`` describe the gateway hop, not this one.
    """
    headers = [("Accept", "*/*"), ("Content-Length", str(len(body)))]
    if content_type:
        headers.append(("Content-Type", content_type))
    return headers


def _render_read_all(up: "_Upstream", backend: Backend) -> bytes:
    """Read a buffered ComfyUI response, as an :class:`UpstreamError` on failure.

    Review finding 4. :func:`open_upstream` wraps only the CONNECT-and-send
    phase; the read happens after it returns, so ComfyUI answering headers and
    then resetting, stalling past the read timeout, or sending a truncated body
    used to let ``OSError``/``http.client.HTTPException`` escape the handler
    entirely — the client lost the facade's documented structured answer and got
    an aborted connection instead.

    Re-raising as ``UpstreamError`` puts the response phase under the SAME
    ``except`` clause in :meth:`_Handler._render_response` that the pre-connection
    failure already took, so every buffered render round trip — submit, poll,
    artifact index, cancel, upload — answers the one retryable 503. The message
    prefix is what keeps that 503 honest about which of the two happened; see
    that handler's body text.
    """
    try:
        return up.read_all()
    except (OSError, http.client.HTTPException, ValueError) as exc:
        raise UpstreamError(f"{backend.name}: failed mid-response: {exc}") from exc


def _render_job_state(payload: object, artifacts: tuple[dict[str, str], ...]) -> str:
    """A coarse, schema-tolerant state for one job: ``completed`` or ``pending``.

    Deliberately NOT a relay of the upstream's own status object. ComfyUI's
    0.33.2 ``/api/jobs`` shape and the older ``/history`` fallback disagree on
    it, and relaying whatever came back would make the facade's contract track
    the upstream's — plus any field it grows later would ship outward
    unreviewed. A caller that needs more detail has the artifact index, which
    is derived, bounded and job-scoped.
    """
    if artifacts:
        return "completed"
    if isinstance(payload, dict):
        status = payload.get("status")
        if isinstance(status, dict) and status.get("completed"):
            return "completed"
    return "pending"


def _json_or_none(raw: bytes) -> object:
    try:
        return json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return None


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
    # Collapsed auth-rejection logging (#228). `None` on every hand-built
    # handler and in the unit suites — every rejection then logs plainly,
    # exactly as it did before this existed.
    rejection_log: RejectionLog | None = None
    # Mesh routes (t6, #237). `None` when mesh is disabled — same treatment as
    # other optional per-server state below.
    mesh_routes: MeshRoutes | None = None
    # The render facade's issued-job map (issue #82, t9). One per server, bound
    # by _make_handler; the class-level instance here is the fallback for a
    # hand-built handler (the unit suites) so the attribute is never None. See
    # RenderJobRegistry for the in-memory/restart contract.
    render_jobs: "RenderJobRegistry" = RenderJobRegistry()
    # The proxied roles' peer specs (proxy-lobes t6, #115/#127), keyed by
    # backend name — built once by peer_specs_from_table and shared with the
    # ReadinessCache's peer-probe thread (see serve). None/empty → the proxy
    # data plane is inert and every request behaves byte-identically to the
    # pre-proxy gateway.
    peer_specs: Mapping[str, PeerSpec] | None = None
    # The replica-pool snapshot provider (t7, #199): backend name → that role's
    # replicas as of the last background probe. None → the pool path is inert
    # and every request behaves byte-identically to the pre-pool gateway. t8
    # binds this to a live ReplicaCache started in serve(); until then no
    # deployment sets it, so the pool ships wired but dormant.
    replica_snapshot: ReplicaSnapshot | None = None
    # The live ReplicaCaches themselves (t8, #199), keyed by backend name —
    # what `replica_snapshot` above reads from, kept separately because
    # GET /capabilities needs the snapshot keyed by ROLE (c9) while dispatch
    # needs it keyed by backend name. None/empty → /capabilities carries no
    # `replicas`/`fingerprint` key at all, exactly as before the pool (h1).
    replica_caches: Mapping[str, ReplicaCache] | None = None
    # The replica-pool in-flight seam (t5, capacity-relative pool routing):
    # counts a dispatch the moment one is placed, so a burst arriving inside
    # one 5 s probe interval does not all read the same idle snapshot. Bound
    # from `replica_caches` by :func:`dispatch_counter`; None → nothing is
    # counted, exactly as the pre-t5 pool behaved.
    dispatch_counter: DispatchCounter | None = None
    # Mesh routing snapshot (W7/W8, mesh-brain). None when mesh is disabled
    # or not yet ready — same treatment as other optional per-server state.
    # Passed to :func:`handle_post` so the mesh dispatch path can select
    # verified members for roles this box lacks (W7) or augment the pool
    # with peers for roles this box hosts (W8).
    mesh_snapshot: "RoutingSnapshot | None" = None
    # HTTP/1.1 so we can stream with chunked transfer encoding.
    protocol_version = "HTTP/1.1"

    # --- inbound auth gate (opt-in, issues #115/#127) ---
    def _authorized(self) -> bool:
        """True when this request may proceed to its route.

        With ``api_key`` unset (auth disabled — the default) this returns
        before the ``Authorization`` header is even READ: no inspection, no
        comparison, so every route is provably byte-identical to the pre-auth
        gateway. With a key set, the credential must be a well-formed
        ``Bearer`` token that matches it timing-safely — see
        :func:`bearer_token_matches`.

        Mesh-enabled servers additionally accept the mesh join key so that
        mesh-authenticated callers can reach data-plane routes without needing
        a separate gateway key.
        """
        api_key = self.server_config.api_key
        if api_key is None:
            return True
        if bearer_token_matches(api_key, self.headers.get("Authorization")):
            return True
        # Mesh join key: when mesh is enabled and the mesh config carries a
        # key, accept it so mesh-authenticated callers can reach data-plane
        # routes (mesh-brain-join t6).
        mr = getattr(self, "mesh_routes", None)
        if mr is not None and mr.config.key is not None:
            return bearer_token_matches(mr.config.key, self.headers.get("Authorization"))
        return False

    def _rejection_source(self) -> str:
        """The peer address to name in the rejection log (#228).

        The SOCKET peer, deliberately — never ``X-Forwarded-For``. That header
        is client-supplied, and a security log an attacker can write the
        attribution field of is worse than one with no attribution: it would
        let a flood be blamed on any address of its choosing. Behind the fleet
        compose bridge this resolves to the calling container or the docker
        gateway, which is exactly the distinction the #228 operator needed
        (in-fleet resident vs. something off-box).
        """
        address = getattr(self, "client_address", None)
        if isinstance(address, tuple) and address:
            return str(address[0])
        return "<unknown>"

    def _log_rejection(self) -> bool:
        """Log this rejection unless it is being collapsed; True if suppressed.

        A suppressed rejection prints NOTHING — not this diagnostic and (via
        :meth:`log_request`) not the ordinary access line either. Printing one
        of the two would have left the observed 1190-line flood at 1190 lines.
        """
        if self.rejection_log is None:
            return False
        line = self.rejection_log.record(
            self._rejection_source(),
            self.command or "?",
            self.path.split("?", 1)[0] or "?",
            rejection_reason(self.headers.get("Authorization")),
        )
        if line is None:
            return True
        sys.stderr.write(f"[gateway] {line}\n")
        return False

    def log_request(self, code="-", size="-") -> None:  # noqa: N802 - stdlib API
        """The ordinary access line, skipped for a collapsed rejection (#228)."""
        if getattr(self, "_rejection_suppressed", False):
            return
        super().log_request(code, size)

    def _reject_unauthorized(self) -> None:
        """Send the 401 ``invalid_api_key`` response and close the connection.

        ``WWW-Authenticate: Bearer`` is the RFC 6750 §3 challenge a 401 to a
        bearer-protected resource must carry. ``Connection: close`` because
        the gate runs BEFORE the request body is read off the socket (a
        rejected request must cost zero parsing and zero upstream sockets):
        leaving an unread body on a kept-alive connection would poison the
        framing of the next request, and ``send_header('Connection',
        'close')`` both advertises and enforces the close
        (``BaseHTTPRequestHandler`` flips ``close_connection`` on it). The
        body/headers never echo any key material — see
        :func:`_invalid_api_key_body`.

        The RESPONSE stays static while the LOG gains a source and a reason
        (#228). That asymmetry is deliberate and is the whole point: the caller
        must not learn whether its key was missing, malformed or merely wrong
        (a 401 must not become a key-material oracle), while the operator
        reading their own stderr needs exactly that to tell a misconfigured
        client from someone guessing. Nothing added to the log travels back.

        The suppression decision is taken BEFORE the response is sent, because
        ``send_response`` is what triggers ``log_request``.
        """
        self._rejection_suppressed = self._log_rejection()
        self._send_simple(
            401,
            [
                ("Content-Type", _CONTENT_TYPE_JSON),
                _WWW_AUTHENTICATE_HEADER,
                ("Connection", "close"),
            ],
            _invalid_api_key_body(),
        )

    # --- GET: /health, /status, /v1/models, /v1/models/supported ---
    def _dispatch_mesh_get(self, route: str) -> bool:
        """Handle ``route`` as a ``GET /mesh/*`` route; ``True`` if it answered.

        Extracted from :meth:`do_GET` (Sonar S3776) — identical behaviour.
        Mesh routes (t6) are gated on the join key by the mesh handler; when
        mesh is disabled no thread starts and every ``/mesh/*`` path falls
        through untouched (returns ``False``) to the normal 404 below.
        """
        if not (_is_mesh_route(route) and self.mesh_routes is not None):
            return False
        result = dispatch_mesh(self, self.mesh_routes)
        if result is not None:
            status, headers, body = result
            self._send_simple(status, headers, body)
            return True
        # Finding 20: mesh enabled but unknown mesh route → 404, not fall-through.
        self._send_simple(
            404,
            [("Content-Type", _CONTENT_TYPE_JSON)],
            json.dumps({"error": {"message": f"not found: {route}", "type": "not_found"}}).encode(),
        )
        return True

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        route = self.path.split("?", 1)[0]
        if self._dispatch_mesh_get(route):
            return
        # Read mesh snapshot once at the top of every request (W2).
        mesh_snapshot = as_routing_snapshot(
            self.mesh_snapshot_holder.current()
            if getattr(self, "mesh_snapshot_holder", None) is not None
            else None
        )
        # Inbound auth (opt-in, #127): the GET /v1/* namespace is DATA PLANE —
        # the model listings are part of the OpenAI surface callers script
        # against. /health, /capabilities and /status stay KEYLESS by design
        # (the container-probe and control-plane surfaces peers must reach
        # before they hold any key) — see the inbound-bearer-auth section
        # above for the full policy.
        if route.startswith("/v1/") and not self._authorized():
            self._reject_unauthorized()
            return
        if is_realtime_path(route):
            # The one GET route that may leave HTTP behind entirely (#149). It
            # sits AFTER the auth gate above by design: the bearer check must
            # cost a rejected handshake zero planning, zero upstream sockets,
            # and zero session state.
            self._handle_realtime(mesh_stt_origin=_first_stt_origin(mesh_snapshot))
        elif route == "/health":
            # `version` is the deployed lobes-cli release THIS gateway process was
            # built from (`__version__`, read off installed package metadata inside
            # the container) — additive, issue #99. It is what lets a remote client
            # (or `lobes doctor`, via lobes.runtime._health.fetch_health) detect
            # deployed-artifact skew docker-free: Dockerfile.gateway pins
            # `pip install "lobes-cli==${MODEL_GEAR_VERSION}"` once, at `lobes init`
            # time, and nothing re-pins it afterwards, so a gateway container can
            # silently run a stale release for days after the host CLI (and PyPI)
            # moved on — exactly what made issue #92 look like a code regression
            # when the fix was already published and simply undeployed.
            self._send_json(
                200, {"status": "ok", "service": "model-gear-gateway", "version": __version__}
            )
        elif route == "/status":
            # Live aggregate the host CLI can't get otherwise: the backends are
            # internal-only, so the gateway fans out to each one's /health + /metrics.
            # The cached pressure sample surfaces the busy-policy state (#85).
            pressure = self.pressure_cache.current() if self.pressure_cache is not None else None
            self._send_json(200, fleet_status_payload(self.table, self.server_config, pressure))
        elif route == "/v1/models":
            self._get_v1_models(mesh_snapshot=mesh_snapshot)
        elif route == "/v1/models/supported":
            # The full catalog of gears you can change to (loaded + the rest),
            # not just the two currently warm. Non-OpenAI shape; /v1/models stays standard.
            self._send_json(200, supported_models_payload(self.table, supported_models_catalog()))
        elif route == "/capabilities":
            self._get_capabilities(mesh_snapshot=mesh_snapshot)
        elif is_render_path(route):
            # The render facade's READ half (issue #82, t9): job status, the
            # job's artifact index, and the artifact bytes. It sits AFTER the
            # `/v1/` bearer gate above, which is the whole reason the family is
            # spelled under /v1/ — ComfyUI behind it has no auth of its own
            # (c2/c26), so an unauthenticated GET reaching here would be served.
            self._deliver(self._render_response(self.path, "GET", b""))
        else:
            self._send_json(404, _not_found_body(route))

    # --- the /v1/render facade (issue #82, t9) -------------------------------
    #
    # ONE entry point for both verbs: the route table is method-aware
    # (:func:`parse_render_route`), so submit/cancel/upload arrive here from
    # do_POST and status/index/artifact from do_GET, and every one of them
    # takes the same infeasibility gate, the same upstream-failure treatment
    # and the same job-scope check.

    def _render_response(self, path: str, method: str, body: bytes) -> GatewayResponse:
        """Answer one request in the render family, or explain why not."""
        route = parse_render_route(path, method)
        if route is None:
            # Not a spelling the allowlist knows — including every ComfyUI
            # surface deliberately left unreachable (/history, /queue, a
            # caller-parameterised /view).
            return _render_json(404, _not_found_body(path.split("?", 1)[0]))
        backend = self._render_backend()
        if backend is None:
            # Hosted nowhere here, or declared infeasible: the same honest 404
            # every dropped role gives, never a half-served lane (#92).
            return GatewayResponse(
                status=404,
                headers=[("Content-Type", _CONTENT_TYPE_JSON)],
                body=_role_infeasible_body(_RENDER_BACKEND, _RENDER_BACKEND),
            )
        handlers = {
            "submit": self._render_submit,
            "status": self._render_status,
            "artifacts": self._render_artifact_index,
            "artifact": self._render_artifact,
            "cancel": self._render_cancel,
            "upload": self._render_upload,
        }
        try:
            return handlers[route.kind](backend, route, body)
        except UpstreamError as exc:
            # The backend is wired but did not complete the round trip. Two
            # distinct phases arrive here and BOTH are this same 503 (review
            # finding 4): the connect-and-send failure open_upstream raises (a
            # cold/stopped ComfyUI, the common case) and the response-phase
            # failure _render_read_all re-raises (headers, then a reset, a stall
            # or a truncated body).
            #
            # Reusing the code rather than minting a second one is deliberate:
            # from the caller's side the two are one fact — the render backend
            # did not answer, wait and retry — and this body is the only place
            # the "lobes never starts it automatically" honesty lives. What the
            # message must NOT do is claim a cold backend it did not observe, so
            # the prose names the mid-response case too and `exc` carries which
            # phase actually failed. Retryable, and never a silent boot: lobes
            # has no lifecycle actuator in the data plane.
            return GatewayResponse(
                status=503,
                headers=[
                    ("Content-Type", _CONTENT_TYPE_JSON),
                    ("Retry-After", str(RENDER_RETRY_AFTER_SECONDS)),
                ],
                body=_render_error_body(
                    f"the render backend did not complete this request ({exc}) — it "
                    "may be cold, still warming up, or it may have dropped the "
                    "connection mid-response. lobes never starts it automatically; "
                    "an operator must bring it up with 'lobes up innereye --apply'. "
                    "Retry shortly.",
                    "render_backend_unavailable",
                ),
            )

    def _render_backend(self) -> Backend | None:
        """This box's wired-and-feasible render tenant, else ``None``."""
        if _RENDER_BACKEND in self.table.infeasible:
            return None
        return next((b for b in self.table.backends if b.name == _RENDER_BACKEND), None)

    def _render_open(
        self,
        backend: Backend,
        path: str,
        *,
        method: str,
        body: bytes = b"",
        ctype: str | None = None,
    ) -> "_Upstream":
        return open_upstream(
            backend,
            path,
            body,
            _render_upstream_headers(body, ctype),
            connect_timeout=self.server_config.connect_timeout,
            read_timeout=self.server_config.read_timeout,
            method=method,
        )

    def _render_fetch(
        self,
        backend: Backend,
        path: str,
        *,
        method: str,
        body: bytes = b"",
        ctype: str | None = None,
    ) -> tuple[int, bytes, list[tuple[str, str]]]:
        """Dial ComfyUI and read the whole answer (never a stream)."""
        up = self._render_open(backend, path, method=method, body=body, ctype=ctype)
        try:
            # _render_read_all, never a bare read_all(): a response-phase
            # failure here must reach _render_response's UpstreamError clause
            # (review finding 4), not escape and abort the client connection.
            return up.status, _render_read_all(up, backend), up.headers
        finally:
            up.close()

    def _render_prompt_id(self, route: RenderRoute) -> str | None:
        return self.render_jobs.prompt_id(route.job_id)

    @staticmethod
    def _render_job_not_found(route: RenderRoute) -> GatewayResponse:
        return _render_error(
            404,
            "render_job_not_found",
            f"no render job `{route.job_id}` was issued by this gateway. Only jobs "
            "submitted through POST /v1/render are addressable here, and a gateway "
            "restart forgets the ids it issued.",
        )

    def _render_submit(self, backend: Backend, route: RenderRoute, body: bytes) -> GatewayResponse:
        """``POST /v1/render`` → ComfyUI ``POST /prompt``, and mint the job id."""
        status, raw, headers = self._render_fetch(
            backend,
            _COMFY_SUBMIT_PATH,
            method="POST",
            body=body,
            ctype=self.headers.get("Content-Type"),
        )
        if status != 200:
            # A rejected graph (ComfyUI's submit-time `node_errors`) is the
            # caller's OWN submission coming back — safe to relay verbatim, and
            # the only way a client learns which node it got wrong. No job id is
            # issued for it.
            return GatewayResponse(status=status, headers=_render_relay_headers(headers), body=raw)
        payload = _json_or_none(raw)
        prompt_id = ""
        if isinstance(payload, dict):
            prompt_id = str(payload.get("prompt_id") or "")
        if not prompt_id:
            return _render_error(
                502,
                "render_submit_failed",
                "the render backend accepted the submission but returned no prompt_id.",
            )
        job_id = self.render_jobs.issue(prompt_id)
        # The response names the GATEWAY's id and nothing of the upstream's —
        # leaking prompt_id would re-open exactly the id space job scoping shuts.
        out: dict = {"job_id": job_id, "status": "queued"}
        if isinstance(payload, dict) and payload.get("node_errors"):
            out["node_errors"] = payload["node_errors"]
        return _render_json(200, out)

    def _render_job_payload(self, backend: Backend, prompt_id: str) -> tuple[int, object]:
        status, raw, _headers = self._render_fetch(
            backend, _COMFY_JOB_PATH.format(quote(prompt_id, safe="")), method="GET"
        )
        return status, _json_or_none(raw)

    def _render_status(self, backend: Backend, route: RenderRoute, body: bytes) -> GatewayResponse:
        """``GET /v1/render/jobs/<job_id>`` → ComfyUI ``GET /api/jobs/<prompt_id>``."""
        prompt_id = self._render_prompt_id(route)
        if prompt_id is None:
            return self._render_job_not_found(route)
        status, payload = self._render_job_payload(backend, prompt_id)
        if status != 200:
            return _render_error(
                502,
                "render_upstream_error",
                f"the render backend answered {status} for this job.",
            )
        artifacts = extract_job_artifacts(payload)
        return _render_json(
            200,
            {
                "job_id": route.job_id,
                "state": _render_job_state(payload, artifacts),
                "artifacts": [dict(a) for a in artifacts],
            },
        )

    def _render_artifact_index(
        self, backend: Backend, route: RenderRoute, body: bytes
    ) -> GatewayResponse:
        """``GET /v1/render/jobs/<job_id>/artifacts`` — this job's outputs only."""
        resp = self._render_status(backend, route, body)
        if resp.status != 200 or resp.body is None:
            return resp
        payload = json.loads(resp.body)
        return _render_json(200, {"job_id": payload["job_id"], "artifacts": payload["artifacts"]})

    def _render_artifact(
        self, backend: Backend, route: RenderRoute, body: bytes
    ) -> GatewayResponse:
        """``GET /v1/render/jobs/<job_id>/artifacts/<name>`` → ``GET /view`` bytes.

        The name is matched against THIS job's own outputs, read back from the
        upstream on every fetch, before ``/view`` is dialed at all. That extra
        round trip is the enforcement: it is what makes the counter-named
        neighbour of a legitimate artifact unreachable rather than merely
        undocumented.
        """
        prompt_id = self._render_prompt_id(route)
        if prompt_id is None:
            return self._render_job_not_found(route)
        status, payload = self._render_job_payload(backend, prompt_id)
        if status != 200:
            return _render_error(
                502,
                "render_upstream_error",
                f"the render backend answered {status} for this job.",
            )
        match = next(
            (a for a in extract_job_artifacts(payload) if a["filename"] == route.artifact),
            None,
        )
        if match is None:
            return _render_error(
                404,
                "render_artifact_not_found",
                "this render job produced no such artifact.",
            )
        query = urlencode(
            {
                "filename": match["filename"],
                "subfolder": match["subfolder"],
                "type": match["type"],
            }
        )
        up = self._render_open(backend, f"{_COMFY_VIEW_PATH}?{query}", method="GET")
        if up.status != 200:
            try:
                raw = _render_read_all(up, backend)
            finally:
                up.close()
            return GatewayResponse(
                status=502,
                headers=[("Content-Type", _CONTENT_TYPE_JSON)],
                body=_render_error_body(
                    f"the render backend answered {up.status} for this artifact "
                    f"({len(raw)} bytes discarded).",
                    "render_upstream_error",
                ),
            )
        # Streamed, never buffered: a render can be tens of megabytes, and
        # `_relay_streaming` re-chunks arbitrary binary verbatim (t8).
        return GatewayResponse(
            status=up.status,
            headers=_render_relay_headers(up.headers),
            upstream=up,
            streaming=True,
        )

    def _render_cancel(self, backend: Backend, route: RenderRoute, body: bytes) -> GatewayResponse:
        """``POST /v1/render/jobs/<job_id>/cancel`` — that job, never the queue.

        Scoped for the same reason innereye's own client refuses to fall back
        to ``POST /interrupt``: interrupting kills whatever is running, which
        is not necessarily the job the caller asked about.
        """
        prompt_id = self._render_prompt_id(route)
        if prompt_id is None:
            return self._render_job_not_found(route)
        status, raw, headers = self._render_fetch(
            backend,
            _COMFY_CANCEL_PATH.format(quote(prompt_id, safe="")),
            method="POST",
            body=b"",
        )
        if status != 200:
            return _render_error(
                502,
                "render_upstream_error",
                f"the render backend answered {status} cancelling this job.",
            )
        return GatewayResponse(status=200, headers=_render_relay_headers(headers), body=raw)

    def _render_upload(self, backend: Backend, route: RenderRoute, body: bytes) -> GatewayResponse:
        """``POST /v1/render/uploads/image`` → ComfyUI ``POST /upload/image``.

        An INPUT, not an output: it writes into ComfyUI's input tree and reads
        nothing back out of the render history, so it needs no job scope. It is
        here because a workflow that starts from an image cannot be submitted
        without it.
        """
        status, raw, headers = self._render_fetch(
            backend,
            _COMFY_UPLOAD_PATH,
            method="POST",
            body=body,
            ctype=self.headers.get("Content-Type"),
        )
        return GatewayResponse(status=status, headers=_render_relay_headers(headers), body=raw)

    # --- GET /v1/realtime: the WebSocket tunnel (issue #149) ---------------
    def _handle_realtime(
        self, mesh_stt_origin: str | None = None
    ) -> None:  # pragma: no cover - opens a socket; see below
        """Tunnel a realtime WebSocket session to the local bridge.

        The refusal paths and the byte pump are unit-tested in
        :mod:`tests.test_gateway_realtime_ws` through
        :mod:`lobes.gateway._realtime`; this method is the socket-owning shell
        that joins them, so it carries the same ``pragma: no cover`` as the
        other socket-level code here.

        On success the handler thread stays parked in :func:`run_tunnel` for
        the whole session and returns only once BOTH directions have unwound —
        which is what makes a dropped client release the thread rather than
        strand it (spec claim c26).
        """
        decision = plan_realtime_upgrade(
            self.table,
            self.server_config,
            self.path,
            list(self.headers.items()),
            mesh_stt_origin=mesh_stt_origin,
        )
        if isinstance(decision, RealtimeRefusal):
            self._refuse_realtime(decision)
            return
        try:
            upstream = socket.create_connection(
                (decision.host, decision.port), timeout=self.server_config.connect_timeout
            )
        except OSError as exc:
            self._send_simple(
                502,
                [("Content-Type", _CONTENT_TYPE_JSON)],
                _error_body("realtime bridge is unavailable", [str(exc)]),
            )
            return
        try:
            # The session is long-lived and mostly idle between utterances, so
            # neither leg may keep the request/response read timeout that HTTP
            # relays use — it would kill a listening session mid-silence.
            upstream.settimeout(None)
            self.connection.settimeout(None)
            upstream.sendall(
                upgrade_request_bytes(
                    decision.path,
                    list(self.headers.items()),
                    host=f"{decision.host}:{decision.port}",
                )
            )
            reader = upstream.makefile("rb")
            try:
                head, leftover = read_head(reader)
            finally:
                reader.detach()  # hand the fd back; the pump owns it from here
        except (OSError, HandshakeError) as exc:
            upstream.close()
            self._send_simple(
                502,
                [("Content-Type", _CONTENT_TYPE_JSON)],
                _error_body("realtime handshake failed", [str(exc)]),
            )
            return
        # Relay the bridge's verdict verbatim — 101 (Sec-WebSocket-Accept and
        # all) or whatever it refused with. Written raw: the handshake is
        # already a complete, correctly framed response, and re-emitting it
        # through send_response/send_header would rewrite it.
        self.close_connection = True
        try:
            self.wfile.write(head)
            self.wfile.flush()
        except OSError:
            upstream.close()
            return
        if status_of(head) != 101:
            upstream.close()
            return
        self.log_message("realtime session opened via %s:%s", decision.host, decision.port)
        try:
            run_tunnel(self.connection, upstream, leftover=leftover)
        finally:
            upstream.close()
            self.log_message("realtime session closed")

    def _refuse_realtime(self, refusal: RealtimeRefusal) -> None:  # pragma: no cover
        """Turn a :class:`RealtimeRefusal` into its response body."""
        if refusal.kind == "role_infeasible":
            self._send_simple(
                404,
                [("Content-Type", _CONTENT_TYPE_JSON)],
                _role_infeasible_body(refusal.role, refusal.role, refusal.peer_origin),
            )
        elif refusal.kind == "audio_not_configured":
            self._send_simple(
                404,
                [("Content-Type", _CONTENT_TYPE_JSON)],
                _error_body("realtime is not configured on this deployment", []),
            )
        else:  # not_an_upgrade
            self._send_simple(
                426,
                [("Content-Type", _CONTENT_TYPE_JSON), ("Upgrade", "websocket")],
                _error_body(
                    "/v1/realtime is a WebSocket route — send an Upgrade: websocket handshake",
                    [],
                ),
            )

    def _get_v1_models(self, *, mesh_snapshot: "RoutingSnapshot | None" = None) -> None:
        # Advertise only backends the live readiness snapshot marks ready
        # (issue #92): a wired-but-dead backend must NOT appear here, so a
        # client can trust that a listed model id reaches a live engine. The
        # snapshot is socket-free (.current() never probes); with no cache
        # wired, every backend is listed (the offline/unit path). A PROXIED
        # role's served id (t6, #115/#127) rides the same rule: the peer
        # spec supplies the id and the snapshot's peer-probe verdict gates
        # it — listed iff the peer verifiably serves it right now.
        ready = self.readiness_cache.current() if self.readiness_cache is not None else None
        peer_served = (
            # Audio peers are excluded here (issue #129): stt/tts are
            # path-routed lanes — their fixed sidecar ids are not requestable
            # via a `model` field, so listing them on /v1/models would invite
            # requests that cannot route. Their honesty surface is
            # /capabilities (hosted_by + proxied + peer-probed ready).
            {
                name: spec.served_name
                for name, spec in self.peer_specs.items()
                if name not in ("stt", "tts")
            }
            if self.peer_specs
            else None
        )
        # A POOLED role is placeable whether or not the operator also armed
        # the singular <PREFIX>_PEER_PROXY — `_peer_only_forward` never
        # consults `peer_proxied`. `peer_specs`, though, is built ONLY from
        # proxied roles, so deriving the served id from it alone would let a
        # pooled-but-unproxied role be placed while /v1/models omitted it,
        # breaking the one-predicate promise this listing exists to keep
        # (Qodo #6 on PR #233). Resolve those names the way
        # `peer_specs_from_table` does, and keep the audio lanes out for the
        # same reason it does: their ids are not requestable via `model`.
        # Finding 13 (review #252): pass the SAME mesh_snapshot /capabilities
        # uses into pooled_backends, so a role available only through a
        # verified mesh member (no declared *_PEER_ORIGINS anywhere) is
        # listed here too, instead of only being placeable at request time.
        pooled_names = pooled_backends(
            self.table,
            replica_snapshot_provider(self.replica_caches),
            mesh_snapshot=mesh_snapshot,
        )
        if pooled_names:
            resolved = dict(peer_served or {})
            for name in pooled_names:
                if name in ("stt", "tts") or resolved.get(name):
                    continue
                served = _peer_served_name(self.table, name, os.environ)
                if served:
                    resolved[name] = served
            peer_served = resolved or None
        # LoRA adapters (hand-lobe plan t4): only those the owning engine's own
        # /v1/models confirmed it loaded. A declared-but-unloaded adapter must
        # never read as usable (#92 for adapters) — see
        # ReadinessCache.current_adapters.
        self._send_json(
            200,
            list_models_payload(
                self.table,
                ready,
                peer_served,
                (
                    self.readiness_cache.current_adapters()
                    if self.readiness_cache is not None
                    else None
                ),
                # A POOLED dropped role is listed on the pool's own evidence
                # (frame decision c17): a caller that never reads /capabilities
                # can still discover a model this box will happily place. The
                # names come from the SAME predicate the placement path uses,
                # and the singular peer probe is not consulted — a pool whose
                # declared peer is down but whose second replica is up is
                # usable, and must therefore be listed.
                pooled=pooled_names,
            ),
        )

    def _get_capabilities(self, *, mesh_snapshot: "RoutingSnapshot | None" = None) -> None:
        # The #81 role→endpoint contract: NINE first-class roles resolved to
        # live metadata via the shared lobes.roles registry. The endpoint is
        # the client-reachable origin this request actually dialed (#87),
        # stt/tts readiness is a live probe of the audio backend (#89), and the
        # five gateway-fronted roles' readiness comes from the background
        # ReadinessCache snapshot (#92) — read socket-free, no probe here.
        cfg = self.server_config
        origin = reachable_origin(self.headers.get("Host"), cfg.public_url)
        audio_ready = probe_audio_ready(cfg.audio_url) is True if cfg.audio_url else None
        # Pass the cache's tri-state snapshot STRAIGHT THROUGH — no boundary
        # coercion here. build_role_registry treats a SUPPLIED backend_ready
        # as authoritative and collapses the cache's None (dead/unreachable)
        # to ready=False itself (issue #92 / honesty h14): coercing the
        # tri-state is the builder's job, not this call site's, so a dead
        # backend can never be advertised ready=True no matter who calls the
        # builder. (This deletes t6's _ready_iff_true bridge — see roles.py.)
        backend_ready = self.readiness_cache.current() if self.readiness_cache is not None else None
        # The CONTEXT half of the peer advert (#220), read from the SAME cache
        # and the same O(1) discipline — a snapshot copy, never a probe.
        peer_context = (
            self.readiness_cache.current_peer_context()
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
                peer_context=peer_context,
                replica_snapshot=replica_role_snapshot(self.replica_caches),
                mesh_snapshot=mesh_snapshot,
            ),
        )

    # --- POST: proxy /v1/* to a backend ---
    def _dispatch_mesh_post(self, route: str) -> bool:
        """Handle ``route`` as a ``POST /mesh/*`` route; ``True`` if it answered.

        Extracted from :meth:`do_POST` (Sonar S3776) — identical behaviour,
        mirroring :meth:`_dispatch_mesh_get` except an unknown mesh route
        answers 405 here (this handler only ever sees POSTs), not 404.
        """
        if not (_is_mesh_route(route) and self.mesh_routes is not None):
            return False
        result = dispatch_mesh(self, self.mesh_routes)
        if result is not None:
            status, headers, body = result
            self._send_simple(status, headers, body)
            return True
        # Finding 20: mesh enabled but unknown mesh route → 405, not fall-through.
        self._send_simple(
            405,
            [("Content-Type", _CONTENT_TYPE_JSON)],
            json.dumps(
                {
                    "error": {
                        "message": f"method not allowed: {self.command} {route}",
                        "type": "method_not_allowed",
                    }
                }
            ).encode(),
        )
        return True

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        route = self.path.split("?", 1)[0]
        if self._dispatch_mesh_post(route):
            return
        # Read mesh snapshot once at the top of every request (W2).
        mesh_snapshot = as_routing_snapshot(
            self.mesh_snapshot_holder.current()
            if getattr(self, "mesh_snapshot_holder", None) is not None
            else None
        )
        # Inbound auth (opt-in, #127): EVERY POST route is data plane — each
        # one is a forward to a backend (chat/completions, completions,
        # embeddings, rerank, score, audio/*). The gate runs before the body
        # is even read off the socket, so a rejected request costs zero body
        # parse, zero model resolution, zero readiness probes (including the
        # audio probe below), and zero upstream connections.
        if not self._authorized():
            self._reject_unauthorized()
            return
        # Per-ROUTE body cap (review finding 1), render-scoped: every other
        # lane's limit is None, so its read is the pre-limit one byte for byte.
        try:
            body = self._read_body(self._post_body_limit())
        except RequestBodyTooLarge as exc:
            self._reject_payload_too_large(exc)
            return
        if is_audio_path(self.path):
            # /v1/audio/* → path-routed, per-ROLE since issue #129 — see
            # handle_audio_request: proxied lane / declared-off referral 404 /
            # the legacy local AUDIO_URL route. The readiness probe (#89) runs
            # only when the LOCAL branch is taken.
            cfg = self.server_config
            resp = handle_audio_request(
                self.table,
                cfg,
                self.peer_specs,
                self.path,
                list(self.headers.items()),
                body,
                open_upstream,
                audio_ready_probe=lambda: (
                    probe_audio_ready(cfg.audio_url) if cfg.audio_url else None
                ),
                mesh_snapshot=mesh_snapshot,
            )
        elif is_render_path(self.path):
            # /v1/render/* → path-routed to the innereye ComfyUI tenant, the
            # exact shape is_audio_path uses one branch up: one backend, no
            # `model` field, no routing table lookup. The WRITE half (submit,
            # cancel, input upload); the read half is in do_GET.
            resp = self._render_response(self.path, "POST", body)
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
                peer_specs=self.peer_specs,
                replica_snapshot=self.replica_snapshot,
                dispatch_counter=self.dispatch_counter,
                mesh_snapshot=mesh_snapshot,
            )
        # The pool's in-flight release (t5) fires HERE, not where the answer
        # was built: a relayed upstream is a one-shot byte tunnel this loop
        # drains long after `_pool_attempt` returned, so the replica is still
        # genuinely busy until the last chunk lands. The outer `finally` makes
        # a mid-relay client disconnect release exactly as a clean completion
        # does — a leaked counter would make this box look permanently full
        # with no way back.
        self._deliver(resp)

    # --- relay helpers ---
    def _deliver(self, resp: GatewayResponse) -> None:
        """Send a built :class:`GatewayResponse`: a gateway body, or a relay.

        Shared by :meth:`do_POST` and :meth:`do_GET` so BOTH verbs get the same
        delivery contract — the same buffered/streaming choice, the same
        upstream close, and the same `release` in a `finally`. Method-agnostic
        by construction: nothing here reads `self.command`.
        """
        try:
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
        finally:
            resp.release()

    def _post_body_limit(self) -> int | None:
        """The maximum request-body size for THIS POST route, or ``None``.

        **Render-scoped by construction** (review finding 1). Every route but
        the ``/v1/render`` family answers ``None``, which makes
        :meth:`_read_body` take exactly the code path it took before limits
        existed — so chat completions, embeddings, rerank/score and the
        ``/v1/audio/*`` multipart lanes are byte-identical. That is the whole
        scoping guarantee, and it is asserted route-by-route in
        tests/test_gateway_render_facade.py rather than argued here.

        Within the family, uploads get their own (much larger) cap and
        everything else — submit, cancel, and any render spelling the allowlist
        will go on to 404 — gets the workflow cap. An unrouted render POST is
        capped deliberately: its body is read BEFORE
        :func:`parse_render_route` refuses it, so leaving it uncapped would
        leave the memory cost exactly where the finding put it.

        A non-positive knob value means "no limit" (the documented escape
        hatch), and surfaces here as ``None`` — the same inert path a
        non-render route takes.
        """
        if not is_render_path(self.path):
            return None
        route = parse_render_route(self.path, "POST")
        cfg = self.server_config
        if route is not None and route.kind == "upload":
            limit = cfg.render_max_upload_bytes
        else:
            limit = cfg.render_max_workflow_bytes
        return limit if limit > 0 else None

    def _post_body_limit_knob(self) -> str:
        """The env key behind :meth:`_post_body_limit` for this route — named in
        the 413 body so the operator is told which of the two caps to raise."""
        route = parse_render_route(self.path, "POST")
        field = (
            "render_max_upload_bytes"
            if route is not None and route.kind == "upload"
            else "render_max_workflow_bytes"
        )
        return RENDER_BODY_LIMIT_ENV[field]

    def _read_body(self, limit: int | None = None) -> bytes:
        """Read the request body, refusing one over ``limit`` before buffering it.

        ``limit=None`` (the default, and what every non-render route passes) is
        the pre-limit behaviour verbatim. With a limit:

        * a ``Content-Length`` over it is refused WITHOUT reading a single byte
          off the socket — the declared size is enough to know;
        * a chunked body is refused incrementally, on the first chunk header
          that would take the total past the cap (see :func:`read_chunked_body`
          with ``strict=True``), since nothing declares its size up front.

        Both raise :class:`RequestBodyTooLarge`, which leaves an unread body on
        the socket — :meth:`_reject_payload_too_large` must therefore close the
        connection rather than keep it alive.
        """
        cl = self.headers.get("Content-Length")
        if cl is not None:
            try:
                length = int(cl)
            except ValueError:
                length = 0
            if limit is not None and length > limit:
                raise RequestBodyTooLarge(length, limit, self._post_body_limit_knob())
            return self.rfile.read(length) if length > 0 else b""
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            if limit is None:
                return read_chunked_body(self.rfile)
            try:
                return read_chunked_body(self.rfile, limit, strict=True)
            except RequestBodyTooLarge as exc:
                # read_chunked_body knows the cap but not whose knob set it.
                raise RequestBodyTooLarge(None, exc.limit, self._post_body_limit_knob()) from exc
        return b""

    def _reject_payload_too_large(self, exc: RequestBodyTooLarge) -> None:
        """Send the 413 and close the connection.

        ``Connection: close`` for the same reason :meth:`_reject_unauthorized`
        carries it: the refusal happens with the oversized body still unread on
        the socket (that is the point — it was never buffered), so keeping the
        connection alive would let those bytes be parsed as the NEXT request's
        head. ``send_header('Connection', 'close')`` both advertises the close
        and makes ``BaseHTTPRequestHandler`` perform it.
        """
        declared = (
            f"declares {exc.declared} bytes, over" if exc.declared is not None else "streamed past"
        )
        self._send_simple(
            413,
            [("Content-Type", _CONTENT_TYPE_JSON), ("Connection", "close")],
            _render_error_body(
                f"this render request body {declared} the gateway's "
                f"{exc.limit}-byte limit for this route, so it was refused without "
                f"being read. Raise {exc.knob} on the gateway (0 disables the cap) "
                "if this deployment genuinely needs larger bodies.",
                "render_payload_too_large",
            ),
        )

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
        """Relay an upstream SSE stream, and ALWAYS end it (issue #220).

        Three ways this returns, and the client can tell them apart:

        * **clean** — the upstream reached EOF; the bytes it sent (its own
          ``data: [DONE]``) are followed by :data:`CHUNK_TERMINATOR`.
        * **upstream died mid-stream** — a read raised; the client gets a
          :func:`sse_error_frame`, then :data:`SSE_DONE`, then the terminator,
          and the connection is closed rather than kept alive. Both frames are
          sent: a caller that parses error events learns *why*, and a caller
          that only watches for the ``[DONE]`` sentinel still terminates.
        * **client hung up** — a write raised; there is nowhere to send a
          terminal frame, so it is logged and the connection closed.

        Before #220 none of the failure paths sent anything at all: the
        exception unwound and left the client blocked on an ESTABLISHED socket
        with no terminal frame (measured: 17-24 minute hangs against an idle
        vLLM). Every exit now writes a terminator or explains why it could not.
        """
        self.send_response(resp.status)
        for key, value in resp.headers:
            self.send_header(key, value)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        upstream_failure: str | None = None
        while True:
            try:
                chunk = resp.upstream.read(_CHUNK)
            except (OSError, http.client.HTTPException, ValueError) as exc:
                # The BACKEND stopped mid-stream (reset, read timeout, a
                # truncated or malformed chunked body). The client is still
                # there and still waiting — tell it.
                #
                # The same triple `open_upstream` catches, and for the same
                # reason: `_Upstream.read` delegates straight to
                # `HTTPResponse.read1`, so a malformed chunk size surfaces as a
                # `ValueError` from the stdlib's own int parse (this module's
                # `read_chunked_body` treats a bad size the same way). Missing
                # it would let the one exception this method exists to handle
                # escape it, skipping the error event, `[DONE]` and the
                # terminator alike — the exact hang #220 is about.
                upstream_failure = f"{type(exc).__name__}: {exc}"
                break
            if not chunk:
                break
            try:
                self.wfile.write(frame_chunk(chunk))
                self.wfile.flush()  # SSE must flush per chunk or it buffers until EOF
            except OSError as exc:
                # The CLIENT went away (BrokenPipeError / ConnectionResetError).
                # Nothing can be delivered to it; stop reading the upstream
                # rather than draining a whole turn into a dead socket. The
                # caller's `finally` still closes the upstream and releases the
                # replica-pool in-flight counter.
                self._abort_stream(resp, f"client disconnected: {type(exc).__name__}: {exc}")
                return
        self._end_stream(resp, upstream_failure)

    def _abort_stream(self, resp: GatewayResponse, reason: str) -> None:
        """Log an undeliverable stream and make sure the socket is not reused."""
        self.close_connection = True
        sys.stderr.write(
            f"[gateway] stream aborted (upstream status {resp.status}, "
            f"attempts {'>'.join(resp.attempts) or '<none>'}): {reason}\n"
        )

    def _end_stream(self, resp: GatewayResponse, upstream_failure: str | None) -> None:
        """Write the terminal frames. Never raises — the response is over either way."""
        tail = CHUNK_TERMINATOR
        if upstream_failure is not None:
            self._abort_stream(resp, f"upstream read failed: {upstream_failure}")
            tail = (
                frame_chunk(
                    sse_error_frame(
                        "the upstream backend stopped sending mid-stream "
                        f"({upstream_failure}); this response is incomplete"
                    )
                )
                + frame_chunk(SSE_DONE)
                + CHUNK_TERMINATOR
            )
        try:
            self.wfile.write(tail)
            self.wfile.flush()
        except OSError as exc:
            # The client hung up between the last chunk and the terminator.
            # There is nothing left to deliver and nothing left to do.
            self._abort_stream(resp, f"client gone before terminator: {type(exc).__name__}: {exc}")

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


# --- the live replica caches (t8, issue #199) -------------------------------
#
# One :class:`~lobes.gateway._replicas.ReplicaCache` per backend, built once at
# process start. Two distinct jobs, deliberately served by the same object:
#
# * **dispatch** reads it by BACKEND name on the request path (O(1), no socket)
#   to place a pooled request (c34/h26);
# * **publication** puts this box's own live fingerprint on GET /capabilities,
#   which is the ONLY way a peer can decide whether our replica is compatible
#   with its own (c33/h25) — the catalog fallback it would otherwise read
#   mislabels an unknown served id (the Orin llama.cpp case).
#
# Because of the second job a cache is built for a hosted lane that declares NO
# peers of its own. Building the CACHE itself is gated purely on
# ``table.replica_origins`` being non-empty ANYWHERE — with it empty (every
# real deployment since t14 deleted the ``<PREFIX>_PEER_ORIGINS`` env parsing
# that used to fill it; only a hand-built table can populate it now) no cache
# is built and no thread is spawned. That used to also mean /capabilities
# carried no ``replicas``/``fingerprint`` key at all (the original h1
# byte-identity guarantee, before a "publish the fingerprint unconditionally"
# reading would have broken it). Review #252 finding 10 punched one hole in
# that: :func:`lobes.roles.annotate_replicas` now publishes an OFFLINE
# (declared, never live-probed) fingerprint for a locally-hosted role even
# with no cache here, whenever the mesh is enabled — a mesh peer has to be
# able to verify a hosted role's fingerprint regardless of whether this box
# also runs an env-declared replica pool for it.

# ``<PREFIX>_<SUFFIX>`` fingerprint suffixes → the lowercase
# :data:`lobes.gateway._replicas.DECLARED_KEYS` names. Only the tool parser's
# env name diverges from a plain lowercasing: the lanes and the compose
# passthrough spell it ``<PREFIX>_TOOL_CALL_PARSER`` (the vLLM flag's own name)
# while the fingerprint field is ``tool_parser``.
_DECLARED_KEY_FOR_SUFFIX: Mapping[str, str] = {"TOOL_CALL_PARSER": "tool_parser"}


def _lane_runtime(backend_name: str, env: Mapping[str, str] | None = None) -> str:
    """The engine a hosted lane runs on ("vllm" / "llamacpp"), from the role
    registry — the same source /capabilities' ``runtime`` field uses.

    Without it a lane that has no declared replica pool is never live-probed,
    every published fingerprint reads ``runtime: unknown``, and the unknown
    rule makes mesh verification impossible (live fleet, 2026-09-12)."""
    role = BACKEND_ROLE.get(backend_name)
    if role is None:
        return ""
    try:
        info = role_registry_from_env(env).get(role)
    except Exception:  # nosec B110 — best-effort: never let an advert helper break wiring
        return ""
    return str(getattr(info, "runtime", "") or "")


def declared_lane_config(lane: Mapping[str, str], *, runtime: str = "") -> dict[str, str]:
    """Adapt one backend's :attr:`RoutingTable.lane_fingerprints` entry to the
    key vocabulary :class:`~lobes.gateway._replicas.LocalLane` expects.

    Any key absent here reads back as ``unknown`` in the published
    fingerprint — never a catalog guess (c33/h25)."""
    declared = {
        _DECLARED_KEY_FOR_SUFFIX.get(suffix, suffix.lower()): value
        for suffix, value in lane.items()
    }
    # A lane's engine is known from the role registry even when nothing ever
    # live-probes it (no declared pool); a declared <PREFIX>_RUNTIME still wins.
    if runtime and not declared.get("runtime"):
        declared["runtime"] = runtime
    return declared


def _check_pool_arming(table: RoutingTable) -> None:
    """Refuse a pool that would silently drop this box's honest referral.

    ``hosted_by`` — the annotation that tells a caller which box actually runs
    a role it asked for here — is read from ``table.peer_origins``, the
    SINGULAR channel (:func:`lobes.roles.annotate_peer_referrals`). The
    plural ``table.replica_origins`` is an ADDITION to it, never a
    replacement, so a table with a plural entry but no matching singular one
    would arm a pool and lose the referral at the same time — and the loss
    would be invisible, because a working pool answers 200 and nobody reads
    ``hosted_by`` until the pool is empty. Refusing at startup is the loud
    version of that bug.

    Retired (t14): both fields used to be populated from the
    ``<PREFIX>_PEER_ORIGINS``/``<PREFIX>_PEER_ORIGIN`` env pair, so this
    guard used to be reachable from a real deployment's ``.env``. That
    parsing is gone — the mesh ``RoutingSnapshot`` (t13) is the pool
    candidate source now, and it never populates these fields either — so
    today this only guards a directly-constructed :class:`RoutingTable`
    (tests, or a future non-env/non-mesh source that fills
    ``replica_origins`` by hand). No operator-facing knob can trigger or fix
    this any more; the names in the raised message describe the table's own
    field convention, not a live env var to set.

    It is also what makes the empty-pool fallback well-defined: "nothing
    selectable falls through to the singular-proxy forward" (frame decision
    c24) presumes a singular origin exists to fall through to.

    Raises :class:`~lobes.gateway._config.ReplicaConfigError`, the same error
    the positional key-length mismatch raises, so a pool misconfiguration is
    one error class end to end.
    """
    # Scoped to DROPPED roles only. A box that HOSTS the role it pools (the
    # #199 case: cortex on the Spark and Thor) publishes no referral at all —
    # `hosted_by` exists precisely for roles this box does not host — so
    # demanding a singular origin there would refuse every pool that shipped
    # before this feature existed. The requirement belongs exactly where the
    # referral does.
    missing = sorted(
        name
        for name in table.replica_origins
        if name in table.infeasible and not table.peer_origins.get(name)
    )
    if not missing:
        return
    # Deferred import: _config imports nothing from here, but the error type
    # belongs to the config layer that owns every other pool parse failure.
    # Retired (t14): PEER_ORIGINS_ENV (the dict this used to look the exact
    # env var name up in) is gone along with the rest of the env peer
    # family — the names below are spelled out directly instead.
    from lobes.gateway._config import ReplicaConfigError

    names = ", ".join(
        f"{name.upper()}_PEER_ORIGINS without {name.upper()}_PEER_ORIGIN" for name in missing
    )
    raise ReplicaConfigError(
        f"{names} — the plural replica channel is an addition to the singular "
        "peer channel, not a replacement: without the singular origin this box "
        "has no hosted_by to publish and nothing to fall back to when no "
        "replica is selectable. Set both fields on the RoutingTable."
    )


def _skips_cache(
    table: RoutingTable, backend: Backend, origins: tuple, declared: Mapping[str, str]
) -> bool:
    """Does this backend get no :class:`ReplicaCache` at all?

    Two independent reasons, split out of :func:`build_replica_caches` to keep
    it under the cognitive-complexity budget (Sonar S3776): a DROPPED lane
    with no declared replicas has nothing to probe or publish, and a
    non-generate lane with neither replicas nor a declared fingerprint has
    nothing to say either.
    """
    if backend.name in table.infeasible and not origins:
        return True
    return not origins and not declared and backend.task != "generate"


def _cache_local_lane(
    table: RoutingTable,
    backend: Backend,
    declared: Mapping[str, str],
    capacities: Mapping[str, float] | None,
) -> "LocalLane | None":
    """This box's own lane for the cache, or ``None`` for a PEER-ONLY pool.

    peer-only-replica-pools, claim c2: a dropped lane has no local replica to
    probe, but its declared peers are still replicas of the same role — so the
    cache is built with ``local=None`` rather than skipped. ``ReplicaCache``
    has always taken an Optional lane; this is the first caller to pass
    ``None``, and ``ReplicaCache._apply_reference`` supplies the compatibility
    reference the missing lane used to be.
    """
    if backend.name in table.infeasible:
        return None
    return LocalLane(
        base_url=backend.base_url,
        served_name=backend.served_name,
        declared=declared_lane_config(declared, runtime=_lane_runtime(backend.name)),
        **_local_weight(capacities, backend.name),
    )


def build_replica_caches(
    table: RoutingTable,
    *,
    urlopen=None,
    start: bool = True,
    capacities: Mapping[str, float] | None = None,
    capacity_kill_switch: bool = False,
) -> dict[str, ReplicaCache]:
    """One refreshed :class:`ReplicaCache` per participating backend, or ``{}``.

    Returns ``{}`` for every deployment that declares no replica pool at all
    (h1). Otherwise every cache is refreshed ONCE
    synchronously, before this returns — :func:`serve` calls this before it
    binds, so the very first request reads a real snapshot instead of the
    unknown seed (the plan's t8 acceptance criterion) — and then handed to its
    daemon threads when ``start``.

    ``urlopen`` is the injected probe seam (the same one
    :class:`ReplicaCache` takes), so this is testable with no sockets.

    ``capacities`` is ``ServerConfig.local_capacities`` — this box's own
    ``<PREFIX>_MAX_ACTIVE`` per backend name (t1) — and is the documented
    carrier for the LOCAL replica's capacity: it reaches ranking as
    :attr:`LocalLane.weight` and needs no new constructor argument of its
    own. A name with no declared capacity keeps the ``UNCALIBRATED_WEIGHT``
    sentinel, which means "nothing published", never "one slot".
    ``capacity_kill_switch`` (``GATEWAY_CAPACITY_KILL_SWITCH``) is handed to
    each cache so a PROBED peer capacity is pinned back to the sentinel too —
    ``_local_capacities`` has already applied it to this box's own numbers,
    so between them the switch holds "local and peer alike", end to end.
    """
    if not table.replica_origins:
        return {}
    _check_pool_arming(table)
    # Deferred import — see the module-level NOTE on the lobes.roles cycle.
    from lobes.roles import BACKEND_ROLE

    role_of = BACKEND_ROLE
    caches: dict[str, ReplicaCache] = {}
    for backend in table.backends:
        origins = table.replica_origins.get(backend.name, ())
        declared = table.lane_fingerprints.get(backend.name, {})
        if _skips_cache(table, backend, origins, declared):
            continue
        keys = table.replica_api_keys.get(backend.name, ())
        caches[backend.name] = ReplicaCache(
            role=role_of.get(backend.name, backend.name),
            local=_cache_local_lane(table, backend, declared, capacities),
            peers=tuple(
                PeerReplica(origin=origin, api_key=keys[i] if i < len(keys) else "")
                for i, origin in enumerate(origins)
            ),
            backend_name=backend.name,
            capacity_kill_switch=capacity_kill_switch,
            urlopen=urlopen,
            start=False,
        )
    for cache in caches.values():
        # A peer-only cache probes ACROSS BOXES on the bind path (c22): a peer
        # that is down, slow, or misbehaving must cost at most the peer probe
        # timeout and must never stop this gateway from serving every role it
        # does host. Each probe already swallows its own failure per-peer; this
        # is the outer guard for anything that escapes the pass entirely.
        try:
            cache.refresh()
        except Exception:  # nosec B110 - boot must not depend on a peer
            pass
        if start:
            cache.start()
    return caches


def replica_snapshot_provider(
    caches: Mapping[str, ReplicaCache] | None,
) -> ReplicaSnapshot | None:
    """The dispatch seam: backend name → that role's replicas, or ``()``.

    ``None`` for an empty/absent cache map, which is what keeps the pool path
    provably inert on a no-pool deployment (:func:`handle_post` short-circuits
    on a ``None`` provider before touching the routing table)."""
    if not caches:
        return None

    def snapshot(backend_name: str) -> tuple[ReplicaState, ...]:
        cache = caches.get(backend_name)
        return cache.current() if cache is not None else ()

    return snapshot


def _local_weight(capacities: Mapping[str, float] | None, name: str) -> dict[str, float]:
    """``{"weight": <declared capacity>}`` for *name*, or ``{}`` (t5).

    An empty dict rather than an explicit sentinel so :class:`LocalLane`'s own
    default is what an undeclared lane gets — one definition of "nothing
    published", not two that can drift.
    """
    declared = (capacities or {}).get(name)
    return {"weight": float(declared)} if declared is not None else {}


def dispatch_counter(
    caches: Mapping[str, ReplicaCache] | None,
) -> DispatchCounter | None:
    """The in-flight seam bound to the live caches, or ``None`` (t5).

    ``None`` for an empty/absent cache map — a deployment with no
    ``*_PEER_ORIGINS`` counts nothing, opens no lock and behaves exactly as
    the pre-pool release does (h1).

    A backend the map does not know yields a no-op release rather than an
    error: dispatch correctness must never depend on the counter being
    present, since the counter is an OPTIMISATION (it makes a burst
    self-correct between probes) and a hard failure here would take down the
    request path it exists to smooth.
    """
    if not caches:
        return None

    def begin(backend_name: str, origin: str) -> "Callable[[], None]":
        cache = caches.get(backend_name)
        if cache is None:
            return _no_release
        token = cache.begin_dispatch(origin)
        return lambda: cache.end_dispatch(token)

    return begin


def replica_role_snapshot(
    caches: Mapping[str, ReplicaCache] | None,
) -> dict[str, tuple[ReplicaState, ...]] | None:
    """The /capabilities seam: ROLE name → that role's replicas (c9).

    ``None`` for an empty/absent cache map, so ``annotate_replicas`` stays a
    no-op and the payload keeps its pre-pool bytes."""
    if not caches:
        return None
    # Deferred import — see the module-level NOTE on the lobes.roles cycle.
    from lobes.roles import ROLE_BACKEND

    return {
        role: caches[backend].current()
        for role, backend in ROLE_BACKEND.items()
        if backend in caches
    }


def _make_handler(
    table: RoutingTable,
    cfg: ServerConfig,
    pressure_cache: PressureCache | None = None,
    readiness_cache: ReadinessCache | None = None,
    peer_specs: Mapping[str, PeerSpec] | None = None,
    replica_snapshot: ReplicaSnapshot | None = None,
    replica_caches: Mapping[str, ReplicaCache] | None = None,
    counter: DispatchCounter | None = None,
    mesh_routes: MeshRoutes | None = None,
    mesh_snapshot_holder: SnapshotHolder | None = None,
) -> type[_Handler]:
    bound = type(
        "_BoundHandler",
        (_Handler,),
        {
            "table": table,
            "server_config": cfg,
            "pressure_cache": pressure_cache,
            "readiness_cache": readiness_cache,
            # One per server, shared across handler threads (#228).
            "rejection_log": RejectionLog(),
            "peer_specs": peer_specs,
            "mesh_routes": mesh_routes,
            "mesh_snapshot_holder": mesh_snapshot_holder,
            # One per server, shared across handler threads — never a global,
            # so two gateways in one process (the test suites) cannot see each
            # other's issued job ids.
            "render_jobs": RenderJobRegistry(),
            # `staticmethod` is load-bearing, not decoration: `replica_snapshot`
            # is the ONLY class attribute here that is a plain function, so it
            # is the only one the descriptor protocol would turn into a BOUND
            # method — `self.replica_snapshot(backend_name)` would then call
            # `snapshot(self, backend_name)` and raise TypeError, taking the
            # whole pooled POST path down with it (t9 caught this live; the
            # unit suites call `handle_post` directly and never see the class).
            "replica_snapshot": (
                None if replica_snapshot is None else staticmethod(replica_snapshot)
            ),
            "replica_caches": replica_caches,
            # `staticmethod` for the same descriptor-protocol reason as
            # `replica_snapshot` above: it is a plain function too.
            "dispatch_counter": (None if counter is None else staticmethod(counter)),
        },
    )
    return bound


def build_mesh_wiring(
    table: RoutingTable,
    cfg: ServerConfig,
    readiness_cache: object | None,
    replica_caches: dict,
    *,
    start: bool = True,
    env: Mapping[str, str] | None = None,
) -> tuple["MeshRoutes | None", "SnapshotHolder | None"]:
    """Build (and by default start) the mesh routes + snapshot holder for serve().

    Returns ``(None, None)`` when the join key is unset — nothing mesh-related
    exists then (the byte-identical contract). Raises ``MeshConfigError`` when
    the key is set but ``GATEWAY_SELF_ORIGIN`` is empty. ``start=False`` builds
    everything without starting the heartbeat thread, for tests.
    """
    mesh_cfg = _build_mesh_config(env)
    if not mesh_cfg.enabled:
        return None, None
    # Finding 1: build a real announcement from gateway data.
    # Finding 7: wire the RejectionLog for flood collapse.
    join_log = RejectionLog()
    # Item C (t9): a second RejectionLog collapses repeated verification
    # failures per-origin, mirroring join_log exactly — a flapping/
    # unreachable peer no longer floods stderr with one line per probe.
    verify_log = RejectionLog()
    mesh_routes, announcement = _build_mesh_routes(
        env=env,
        self_origin=_require_self_origin(table.self_origin),
        readiness_cache=readiness_cache,
        replica_caches=replica_caches,
        local_capacities=cfg.local_capacities,
        # NOTE (task t11, issue #92 c9/h20): the render lane (`innereye`) is
        # deliberately NOT filtered out of this dict — `declared_lane_configs`
        # feeds only the lower-level `_build_announcement` entry point (its
        # own `MESH_UNFORWARDABLE_ROLES` guard, in `_mesh_routes.py`, drops
        # it there too). The announcement this box actually heartbeats is
        # `_capabilities_announcement` below, built from `GET /capabilities`
        # via `announcement_from_capabilities` — see
        # `_role_info_from_capability_entry`'s matching guard, the path that
        # matters live. Both guards live beside `MESH_UNFORWARDABLE_ROLES`
        # itself in `lobes/roles.py`.
        declared_lane_configs={
            b.name: declared_lane_config(
                table.lane_fingerprints.get(b.name, {}), runtime=_lane_runtime(b.name, env)
            )
            for b in table.backends
        },
        join_log=join_log,
        verify_log=verify_log,
        missed_max=mesh_cfg.missed_max,
    )
    # Finding 11 (review #252): the request handlers and the heartbeat must
    # read/write ONE holder. `_build_mesh_routes` may already have attached
    # one to `mesh_routes._holder`; reuse it, and seed it BEFORE the heartbeat
    # thread starts, so the thread never captures a stale or empty instance.
    holder = mesh_routes._holder or SnapshotHolder(mesh_routes.roster)  # noqa: SLF001
    mesh_routes._holder = holder  # noqa: SLF001
    holder.replace(
        MeshRoutingView(snapshot=build_snapshot(mesh_routes.roster), peer_states={}),
    )

    # The announcement IS this box's own /capabilities payload, filtered to the
    # roles it hosts — so what a member announces and what a peer reads back
    # when verifying are the same bytes by construction. Rebuilt on every
    # POST /mesh/reannounce (lobes switch / lobes up / a lane going unhealthy)
    # from the live readiness view (c46/h37).
    def _capabilities_announcement() -> Announcement:
        ready = None
        try:
            ready = readiness_cache.current() if readiness_cache is not None else None
        except (
            Exception
        ):  # nosec B110 — best-effort: an unreadable cache announces without readiness
            ready = None
        # Same inputs GET /capabilities uses for the fingerprint: the live
        # replica snapshot when a lane is probed, the offline one otherwise —
        # so announced and advertised fingerprints are identical bytes.
        payload = capabilities_payload(
            table,
            cfg,
            env,
            backend_ready=ready,
            replica_snapshot=replica_role_snapshot(replica_caches),
            mesh_snapshot=build_snapshot(mesh_routes.roster),
        )
        return announcement_from_capabilities(
            mesh_cfg,
            payload,
            self_origin=table.self_origin,
            local_capacities=cfg.local_capacities,
        )

    announcement = _capabilities_announcement()
    mesh_routes.set_announcement_builder(_capabilities_announcement)
    if start:
        # Start the heartbeat daemon thread after the holder is seeded.
        _start_mesh(mesh_routes, announcement)
    return mesh_routes, holder


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
    #
    # Proxied roles (t6, #115/#127): ONE PeerSpec set is built here from the
    # routing table (origin + resolved served id + pairwise key per proxied
    # name) and shared by BOTH consumers — the cache's peer-probe thread (so
    # /v1/models + /capabilities advertise a proxied role only while its peer
    # verifiably serves the id) and the handler (so the data-plane forward
    # dials exactly what the probe verified). No proxy config → empty specs →
    # no peer thread, no proxy branch, byte-identical pre-proxy behaviour.
    peer_specs = peer_specs_from_table(table)
    # LoRA-bearing backends (hand-lobe plan t4): the cache additionally probes
    # each one's OWN /v1/models to learn which declared adapters the engine
    # actually loaded. Empty for every backend with no declared adapters — i.e.
    # every deployment that has not declared HAND_LORA_MODULES — so this adds
    # no probe traffic and changes nothing until an operator declares one.
    adapter_targets = {b.name: (b.base_url, b.adapters) for b in table.backends if b.adapters}
    # The render lane's readiness path override (see ``_COMFY_READY_PATH``
    # above) — the ONLY backend that gets a non-``/health`` entry; every
    # other lane is unaffected (`paths.get(name, "/health")` inside
    # ReadinessCache falls through to the shared default).
    readiness_paths = {
        b.name: _COMFY_READY_PATH for b in table.backends if b.name == _RENDER_BACKEND
    }
    readiness_cache = ReadinessCache.from_backends(
        table.backends,
        peer_specs=tuple(peer_specs.values()),
        adapter_targets=adapter_targets,
        paths=readiness_paths,
        start=False,
    )
    readiness_cache.refresh()
    readiness_cache.start()
    # The replica pool (t8, #199). Built, refreshed ONCE synchronously, and
    # started BEFORE binding, so the first request placed by the pool reads a
    # real snapshot rather than the unknown seed — the same
    # refresh-then-start discipline the readiness cache above uses, and for
    # the same reason. Empty dict (no *_PEER_ORIGINS declared anywhere) → no
    # threads, a None snapshot provider, and a byte-identical gateway (h1).
    replica_caches = build_replica_caches(
        table,
        capacities=cfg.local_capacities,
        capacity_kill_switch=cfg.capacity_kill_switch,
    )
    # Mesh (t6): build the mesh routes when the join key is set; when
    # disabled (no key) every /mesh/* path falls through to the 404 below.
    # The wiring lives in build_mesh_wiring() so a test can exercise it with a
    # mesh-enabled config without binding a socket — serve() itself is
    # `pragma: no cover`, which is how a `cfg.self_origin` typo (the attribute
    # lives on the RoutingTable) reached a live box on 2026-09-12.
    mesh_routes, holder = build_mesh_wiring(table, cfg, readiness_cache, replica_caches)
    httpd = ThreadingHTTPServer(
        (cfg.host, cfg.port),
        _make_handler(
            table,
            cfg,
            pressure_cache,
            readiness_cache,
            peer_specs,
            replica_snapshot_provider(replica_caches),
            replica_caches,
            dispatch_counter(replica_caches),
            mesh_routes,
            mesh_snapshot_holder=holder if mesh_routes is not None else None,
        ),
    )
    sys.stderr.write(f"[gateway] listening on {cfg.host}:{cfg.port}\n")
    try:
        httpd.serve_forever()
    finally:
        # Bounded, idempotent, and daemon-backed either way — a straggler probe
        # never blocks exit. Mirrors ReadinessCache's own stop() contract.
        for cache in replica_caches.values():
            cache.stop()
        # Stop the mesh heartbeat thread when the server exits.
        if mesh_routes is not None:
            mesh_routes._stop.set()  # noqa: SLF001
            if mesh_routes._thread is not None:  # noqa: SLF001
                mesh_routes._thread.join(timeout=3)  # noqa: SLF001
