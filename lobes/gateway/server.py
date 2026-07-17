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
role whose operator armed ``<PREFIX>_PEER_PROXY`` (``table.peer_proxied``) is
answered by FORWARDING the request to the operator-declared peer origin,
replacing the referral 404 for exactly those names and nothing else. See the
"proxy data plane" section below for the loop guard, the pairwise-credential
swap, the failure modes, and why proxied requests bypass the LOCAL pressure
policy.
"""

from __future__ import annotations

import dataclasses
import hmac
import http.client
import json
import os
import re
import sys
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Iterable
from urllib.parse import urlsplit

from lobes import __version__, _metrics
from lobes.catalog import SUPPORTED_MODELS
from lobes.catalog import as_dicts as supported_models_catalog
from lobes.gateway._config import ServerConfig
from lobes.gateway._pressure_policy import BUSY_RETRY_AFTER_SECONDS, decide
from lobes.gateway._readiness import PeerSpec, ReadinessCache
from lobes.gateway._routing import (
    Backend,
    RoutingTable,
    audio_role_for_path,
    infeasible_owner,
    is_audio_path,
    is_unknown_model,
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
    requested: str | None, backend_name: str, peer_origin: str | None = None
) -> bytes:
    """4xx body for a request pinned to a HARDWARE-infeasible backend (t6).

    Distinct ``type``/``code`` from :func:`_model_not_found_body`: the
    requested id/role IS part of the six-role contract (it may even be
    wired — the primary is unconditionally wired regardless of feasibility)
    but this machine's per-machine profile declared its owning backend
    (``backend_name``) unable to serve it at all. Never a reason to
    silently substitute a different, feasible gear — see
    :func:`lobes.gateway._routing.infeasible_owner`.

    ``peer_origin`` is the opt-in honest referral (mesh-brain t3, issue
    #112): the OPERATOR-DECLARED origin of the peer box that hosts this role
    (:data:`lobes.gateway._config.PEER_ORIGIN_ENV`). When set, the message
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
    return json.dumps({"error": error}).encode("utf-8")


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
# (``table.infeasible``) whose operator declared a peer origin AND armed the
# ``<PREFIX>_PEER_PROXY`` knob (``table.peer_proxied``, t1) is answered by
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

# 508 Loop Detected — the refusal for a marked request that would re-proxy.
_PROXY_LOOP_STATUS = 508

# Backend name → the deployment env var carrying that role's served model id
# (the same ``<PREFIX>_SERVED_NAME`` convention build_config reads). Consulted
# by _peer_served_name for a proxied role whose backend is UNWIRED locally.
_PEER_SERVED_NAME_ENV: dict[str, str] = {
    "primary": "PRIMARY_SERVED_NAME",
    "multimodal": "MULTIMODAL_SERVED_NAME",
    "muse": "MUSE_SERVED_NAME",
    "embed": "EMBED_SERVED_NAME",
    "rerank": "RERANK_SERVED_NAME",
}

# Backend name → the catalog ``role_hint`` of its canonical model — the same
# fallback lobes.roles uses to NAME an unwired role's model. Scoped, like every
# peer channel, to the five core roles (see lobes.gateway._config.PEER_PROXY_ENV).
_PEER_ROLE_HINT: dict[str, str] = {
    "primary": "primary",
    "multimodal": "multimodal",
    "muse": "muse",
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
    if name in ("stt", "tts"):
        # First-class audio roles (issue #129): fixed sidecar checkpoints, not
        # catalog gears — the id is the SAME constant lobes.roles advertises on
        # /capabilities (lazy import: matches capabilities_payload's own
        # deferred lobes.roles import below).
        from lobes.roles import _STT_MODEL, _TTS_MODEL

        return _STT_MODEL if name == "stt" else _TTS_MODEL
    wired = next((b.served_name for b in table.backends if b.name == name), None)
    if wired:
        return wired
    from_env = (env.get(_PEER_SERVED_NAME_ENV.get(name, "")) or "").strip()
    if from_env:
        return from_env
    hint = _PEER_ROLE_HINT.get(name)
    return next((m.id for m in SUPPORTED_MODELS if m.role_hint == hint), "")


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


def _arriving_hop_marker(req_headers: Iterable[tuple[str, str]]) -> str | None:
    """The inbound ``X-Lobes-Proxied`` marker value, if the request carries one."""
    marker_key = PROXIED_HEADER.lower()
    for key, value in req_headers:
        if key.lower() == marker_key:
            return value
    return None


def _proxy_loop_body(arriving: str, spec: PeerSpec) -> bytes:
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


def _peer_unavailable_response(spec: PeerSpec, attempts: list[str]) -> GatewayResponse:
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
    )


def _peer_declined_body(spec: PeerSpec, raw: bytes) -> bytes | None:
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
    spec: PeerSpec,
    path: str,
    req_headers: Iterable[tuple[str, str]],
    body: bytes,
    open_upstream: OpenUpstream,
    *,
    rewrite: bool = True,
) -> GatewayResponse:
    """Forward one request to its proxied role's peer and relay the outcome.

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
    arriving = _arriving_hop_marker(req_headers)
    if arriving is not None:
        # Single-hop guard: this request was already forwarded once by a peer
        # gateway; departing again could ping-pong between misconfigured boxes
        # forever. Refuse with zero outbound attempts. (A marked arrival whose
        # role is served LOCALLY never reaches this function — see handle_post.)
        return GatewayResponse(
            status=_PROXY_LOOP_STATUS,
            headers=[("Content-Type", _CONTENT_TYPE_JSON)],
            body=_proxy_loop_body(arriving, spec),
        )
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
        attempts = [f"{peer_backend.name}: HTTP {up.status}"]
        up.close()
        return _peer_unavailable_response(spec, attempts)
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
        return GatewayResponse(status=404, headers=[proxied_by] + up.headers, body=raw)
    # 2xx or any other 4xx: the peer's authoritative verdict, relayed exactly
    # like the single-owner rules relay a local backend's (#91) — including
    # the peer's own 429 pressure shed riding back to the caller.
    return GatewayResponse(
        status=up.status,
        headers=[proxied_by] + up.headers,
        upstream=up,
        streaming=streaming,
    )


def _feasibility_response(table: RoutingTable, requested: str | None) -> GatewayResponse | None:
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
    # Opt-in honest referral (mesh-brain t3): when the operator declared the
    # peer that hosts this role (table.peer_origins), the 404 names it — as an
    # ANNOTATION only. The request is still answered HERE, terminally; a
    # referral-only role is never forwarded (the proxy data plane, t6, only
    # fires for names in table.peer_proxied, which handle_post routes to
    # _proxy_to_peer BEFORE this gate — so every 404 built here stays
    # byte-identical to the pre-proxy contract). No declaration → the
    # pre-referral body, byte for byte.
    return GatewayResponse(
        status=404,
        headers=[("Content-Type", _CONTENT_TYPE_JSON)],
        body=_role_infeasible_body(
            requested, infeasible_name, table.peer_origins.get(infeasible_name)
        ),
    )


def _resolve_tier(
    table: RoutingTable,
    requested: str | None,
    pressure: dict[str, float],
    override: bool,
) -> tuple[GatewayResponse | None, str | None, list[tuple[str, str]]]:
    """The tier-alias branch of :func:`handle_post`: hardware feasibility gate,
    then pressure-aware busy shedding (#85), then the resolved served name.

    Returns ``(early_response, served, tier_headers)``. When ``early_response``
    is not ``None`` the caller must return it immediately without dialing any
    backend; ``served``/``tier_headers`` are only meaningful otherwise.
    """
    early = _feasibility_response(table, requested)
    if early is not None:
        return early, None, []
    decision = resolve_tier_request(requested, pressure, override, table)
    if decision["busy"]:
        busy_response = GatewayResponse(
            status=429,
            headers=[
                ("Retry-After", str(BUSY_RETRY_AFTER_SECONDS)),
                ("X-Lobes-Tier-Reason", "busy"),
                ("Content-Type", _CONTENT_TYPE_JSON),
            ],
            body=_busy_body(decision["requested_tier"]),
        )
        return busy_response, None, []
    served = decision["served_name"]
    tier_headers = [
        ("X-Lobes-Tier", decision["served_tier"]),
        ("X-Lobes-Tier-Reason", decision["reason"]),
    ]
    return None, served, tier_headers


def _resolve_plain_model(
    table: RoutingTable, requested: str | None
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
    early = _feasibility_response(table, requested)
    if early is not None:
        return early, None
    return None, resolve_model(table, requested)


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


def _try_primary_with_strict_retry(
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

    Force-strict-tools (opt-in, colleague#320): only the primary/cortex lane,
    only chat-completions, only a body an injection actually changed. Every
    other request takes the untouched :func:`_try_backends` call below — this
    is the byte-identical-passthrough guarantee when the knob is off (or
    simply inapplicable to this request). Extracted from :func:`handle_post`
    verbatim (Sonar S3776); returns exactly what the dial helpers return:
    ``(response-or-None, attempts)``.
    """
    strict_injection = None
    if (
        cfg.force_strict_tools
        and ordered[0].name == "primary"
        and _is_chat_completions_request(path)
    ):
        strict_injection = inject_strict_tools(fwd_body)

    if strict_injection is not None:
        injected_body, tool_names = strict_injection
        return _try_primary_with_strict_retry(
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
    to be served instead of shed. On the served path the ``X-Lobes-Tier`` /
    ``X-Lobes-Tier-Reason`` headers still travel with the response (prepended,
    streaming-safe). A plain model id, or ``pressure=None``, takes the existing
    non-tier path unchanged.

    Force-strict-tools (``cfg.force_strict_tools``, opt-in, colleague#320): once
    the owner is resolved, a ``/v1/chat/completions`` request whose owner is the
    ``primary`` (cortex) backend and whose body carries a non-empty ``tools``
    array is passed through :func:`inject_strict_tools`. When that ACTUALLY
    modifies the body (at least one ``tools[i].function`` lacked ``strict``),
    dialing is handed to :func:`_try_primary_with_strict_retry` instead of
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
    """
    requested = extract_model(body)
    if peer_specs:
        proxied_name = _proxied_owner(table, peer_specs, requested)
        if proxied_name is not None:
            return _proxy_to_peer(
                cfg, peer_specs[proxied_name], path, req_headers, body, open_upstream
            )
    tier_headers: list[tuple[str, str]] = []
    if pressure is not None and is_tier_alias(requested):
        # Hardware feasibility gate (issue #92 extended to the HARDWARE
        # dimension, task t6) runs BEFORE pressure-shedding/upward-fallback: an
        # infeasible role is an absolute hardware fact, not a load condition, so
        # it takes priority over — and is never bypassed by — X-Lobes-Override.
        # Checked on the LITERAL requested tier so an explicitly-named
        # infeasible role (e.g. "cortex") is rejected outright, never silently
        # re-routed to a different, feasible gear via the tier system's normal
        # upward-fallback substitution.
        early, served, tier_headers = _resolve_tier(table, requested, pressure, override)
        if early is not None:
            return early
    else:
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
        early, served = _resolve_plain_model(table, requested)
        if early is not None:
            return early
    streaming = is_streaming(body)
    fwd_body = rewrite_model(body, served)
    fwd_headers = filter_headers(req_headers)

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

    response, attempts = _dial_owner(
        ordered, cfg, path, fwd_body, fwd_headers, open_upstream, streaming, tier_headers
    )
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
        return GatewayResponse(
            status=404,
            headers=[("Content-Type", _CONTENT_TYPE_JSON)],
            body=_role_infeasible_body(role, role, table.peer_origins.get(role)),
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


def capabilities_payload(
    table: RoutingTable,
    cfg: ServerConfig,
    env: Mapping[str, str] | None = None,
    *,
    gateway_url: str | None = None,
    audio_ready: bool | None = None,
    backend_ready: Mapping[str, bool | None] | None = None,
) -> dict:
    """The seven first-class roles (issue #81), resolved via the shared registry.

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
    """
    # deferred imports — see the module-level NOTE
    from lobes.roles import ROLES, annotate_peer_referrals, build_role_registry

    resolved_env = os.environ if env is None else env
    peer_ready = None
    if backend_ready is not None and table.peer_proxied:
        peer_ready = {name: backend_ready.get(name) for name in table.peer_proxied}
    registry = build_role_registry(
        table,
        cfg,
        env=resolved_env,
        gateway_url=gateway_url,
        audio_ready=audio_ready,
        backend_ready=backend_ready,
        peer_ready=peer_ready,
    )
    payload = {role: dataclasses.asdict(registry[role]) for role in ROLES}
    # Opt-in honest referral (mesh-brain t3): annotate each unhosted
    # (feasible=false) role with the OPERATOR-DECLARED peer origin that hosts
    # it (table.peer_origins). With no peer config (the default) this is a
    # no-op and the payload stays byte-identical to the pre-referral contract.
    return annotate_peer_referrals(payload, table)


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
    # The proxied roles' peer specs (proxy-lobes t6, #115/#127), keyed by
    # backend name — built once by peer_specs_from_table and shared with the
    # ReadinessCache's peer-probe thread (see serve). None/empty → the proxy
    # data plane is inert and every request behaves byte-identically to the
    # pre-proxy gateway.
    peer_specs: Mapping[str, PeerSpec] | None = None
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
        """
        api_key = self.server_config.api_key
        if api_key is None:
            return True
        return bearer_token_matches(api_key, self.headers.get("Authorization"))

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
        """
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
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        route = self.path.split("?", 1)[0]
        # Inbound auth (opt-in, #127): the GET /v1/* namespace is DATA PLANE —
        # the model listings are part of the OpenAI surface callers script
        # against. /health, /capabilities and /status stay KEYLESS by design
        # (the container-probe and control-plane surfaces peers must reach
        # before they hold any key) — see the inbound-bearer-auth section
        # above for the full policy.
        if route.startswith("/v1/") and not self._authorized():
            self._reject_unauthorized()
            return
        if route == "/health":
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
            self._get_v1_models()
        elif route == "/v1/models/supported":
            # The full catalog of gears you can change to (loaded + the rest),
            # not just the two currently warm. Non-OpenAI shape; /v1/models stays standard.
            self._send_json(200, supported_models_payload(self.table, supported_models_catalog()))
        elif route == "/capabilities":
            self._get_capabilities()
        else:
            self._send_json(404, {"error": {"message": f"not found: {route}", "type": "not_found"}})

    def _get_v1_models(self) -> None:
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
        self._send_json(200, list_models_payload(self.table, ready, peer_served))

    def _get_capabilities(self) -> None:
        # The #81 role→endpoint contract: SEVEN first-class roles resolved to
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

    # --- POST: proxy /v1/* to a backend ---
    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        # Inbound auth (opt-in, #127): EVERY POST route is data plane — each
        # one is a forward to a backend (chat/completions, completions,
        # embeddings, rerank, score, audio/*). The gate runs before the body
        # is even read off the socket, so a rejected request costs zero body
        # parse, zero model resolution, zero readiness probes (including the
        # audio probe below), and zero upstream connections.
        if not self._authorized():
            self._reject_unauthorized()
            return
        body = self._read_body()
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
                peer_specs=self.peer_specs,
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
    peer_specs: Mapping[str, PeerSpec] | None = None,
) -> type[_Handler]:
    bound = type(
        "_BoundHandler",
        (_Handler,),
        {
            "table": table,
            "server_config": cfg,
            "pressure_cache": pressure_cache,
            "readiness_cache": readiness_cache,
            "peer_specs": peer_specs,
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
    #
    # Proxied roles (t6, #115/#127): ONE PeerSpec set is built here from the
    # routing table (origin + resolved served id + pairwise key per proxied
    # name) and shared by BOTH consumers — the cache's peer-probe thread (so
    # /v1/models + /capabilities advertise a proxied role only while its peer
    # verifiably serves the id) and the handler (so the data-plane forward
    # dials exactly what the probe verified). No proxy config → empty specs →
    # no peer thread, no proxy branch, byte-identical pre-proxy behaviour.
    peer_specs = peer_specs_from_table(table)
    readiness_cache = ReadinessCache.from_backends(
        table.backends, peer_specs=tuple(peer_specs.values()), start=False
    )
    readiness_cache.refresh()
    readiness_cache.start()
    httpd = ThreadingHTTPServer(
        (cfg.host, cfg.port),
        _make_handler(table, cfg, pressure_cache, readiness_cache, peer_specs),
    )
    sys.stderr.write(f"[gateway] listening on {cfg.host}:{cfg.port}\n")
    httpd.serve_forever()
