"""End-to-end integration for proxy-lobes + pairwise auth (t8, issues #115/#127).

TWO REAL GATEWAYS TALKING OVER REAL SOCKETS — the cross-cutting layer the
per-task suites (tests/test_gateway_config_proxy.py, test_gateway_auth.py,
test_readiness_peer_probe.py, test_roles_proxied.py, test_gateway_proxy.py)
deliberately don't cover. Nothing here monkeypatches ``open_upstream``: every
hop is a genuine HTTP exchange.

The harness (see :func:`_two_gateways`):

* a **PEER box** — a real gateway (``ThreadingHTTPServer`` +
  ``S._make_handler``) built by ``build_config`` from a thor-lobe-shaped env
  (cortex dropped, ``senses`` hosted at a fake vLLM backend), with its OWN
  inbound key (``GATEWAY_API_KEY``) and a real, ``refresh()``-seeded
  :class:`ReadinessCache`. Every request it receives (probe GETs and forwarded
  POSTs alike) is captured by a recording handler subclass.
* a **PROXYING box** — a real gateway built from a spark-lobe-shaped env
  (``MULTIMODAL_FEASIBLE=false`` + ``MULTIMODAL_PEER_ORIGIN=<the peer's live
  loopback origin>`` + ``MULTIMODAL_PEER_PROXY=true`` +
  ``MULTIMODAL_PEER_API_KEY=<the peer's inbound key>`` +
  ``GATEWAY_API_KEY=<its own, different inbound key>``), its local roles
  wired to a second fake backend, with its own real ReadinessCache whose peer
  probe dials the live peer gateway.

Coverage map (spec claim c14 — one test per after-state element):

* (a) a chat request naming ``senses``/``multimodal`` against the proxying
  box with ITS key → 200, the peer backend's body, the peer's served id,
  ``X-Lobes-Proxied-By`` = the declared origin verbatim;
* (b) missing/wrong key → 401 from the proxying box and the peer NEVER saw a
  request (asserted on the peer's inbound request log);
* (c) ``GET /capabilities`` shows ``proxied: true`` + ``hosted_by`` + live
  ``ready``; peer stopped + refresh → ready honest and the proxied id GONE
  from ``/v1/models``;
* (d) a request pre-marked ``X-Lobes-Proxied`` that would depart again → 508
  ``proxy_loop``, peer log unchanged;
* (e) peer down on the data plane → 503 + ``Retry-After``;
* (f) peer declines 404 ``role_infeasible`` → terminal relay naming the peer,
  exactly one outbound attempt.

Plus: credential hygiene end-to-end (the pairwise key on EVERY request the
peer received, the caller's key on NONE — and neither key anywhere the client
or the logs can see), SSE relayed incrementally across both hops, and the h7
byte-identical no-config goldens (/capabilities, /v1/models, and the
role_infeasible 404 pinned as literal expected JSON for a deployment with no
proxy/auth/peer knobs — cross-checked against the shapes the pre-feature
suites pin, e.g. tests/test_roles_proxied.py's oracle tests and
tests/test_gateway_proxy.py's referral-404 test).
"""

from __future__ import annotations

import contextlib
import http.client
import io
import json
import os
import threading
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import monotonic as _monotonic
from types import SimpleNamespace
from unittest import mock
from urllib.parse import urlsplit

import pytest

from lobes.gateway import server as S
from lobes.gateway._config import build_config
from lobes.gateway._readiness import ReadinessCache
from lobes.gateway._selection import (
    REASON_AFFINITY,
    REASON_LOCAL_BUSY_FORWARDED,
    REASON_LOCAL_IDLE,
    REASON_PEER_LESS_LOADED,
    REASON_SOLE_READY,
)

_CORTEX_ID = "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"
_SENSES_ID = "coolthor/gemma-4-12B-it-NVFP4A16"  # the catalog multimodal default
_EMBED_ID = "Qwen/Qwen3-Embedding-0.6B"
_RERANK_ID = "Qwen/Qwen3-Reranker-0.6B"

# The PAIRWISE key: the peer box's inbound GATEWAY_API_KEY, handed to the
# proxying box as its outbound MULTIMODAL_PEER_API_KEY. Distinct from...
_PEER_KEY = "sk-pairwise-peer-inbound-7401"
# ...the proxying box's OWN inbound key — the one its callers use. It must
# never travel to the peer (nor leak anywhere a client or log can see).
_CALLER_KEY = "sk-proxying-box-inbound-2233"

# A loopback port with nothing listening — the peer box's dropped-cortex URL
# refuses instantly instead of hanging a readiness refresh on DNS.
_CLOSED_URL = "http://127.0.0.1:9"

_MODELS_PATH = "/v1/models"
_METRICS_PATH = "/metrics"
_CHAT_PATH = "/v1/chat/completions"

_SSE_EVENTS = (
    b'data: {"choices":[{"delta":{"content":"pro"}}]}\n\n',
    b'data: {"choices":[{"delta":{"content":"xy"}}]}\n\n',
    b"data: [DONE]\n\n",
)


def _chat_completion_body(served_id: str) -> dict:
    """The canned OpenAI-shaped answer the fake vLLM backend produces."""
    return {
        "id": f"chatcmpl-fake-{served_id}",
        "object": "chat.completion",
        "model": served_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": f"answered-by-{served_id}"},
                "finish_reason": "stop",
            }
        ],
    }


def _serve_in_thread(httpd) -> None:
    """Run *httpd* on a daemon thread with a short poll interval, so a test's
    ``shutdown()`` returns promptly instead of waiting the default 0.5s poll."""
    threading.Thread(target=lambda: httpd.serve_forever(poll_interval=0.02), daemon=True).start()


# --- the fake vLLM engine (a real HTTP server) -------------------------------


class _FakeBackendHandler(BaseHTTPRequestHandler):
    """GET /health → 200; POST → a canned chat completion, or an SSE stream
    when the body asks ``"stream": true``. Records every request it receives.
    HTTP/1.0 (the class default) so the SSE response is close-delimited and the
    relaying gateway's ``read1`` loop sees frames as they are flushed."""

    def log_message(self, *_args) -> None:  # keep stderr for the gateways' logs
        pass

    def _record(self, body: bytes = b"") -> None:
        self.server.log.append(
            SimpleNamespace(
                method=self.command,
                path=self.path,
                headers=list(self.headers.items()),
                body=body,
            )
        )

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status: int, text: str) -> None:
        body = text.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._record()
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        elif self.path == _MODELS_PATH:
            # The vLLM-shaped model list the LOCAL replica probe fingerprints
            # off (served id + max_model_len live, runtime from `owned_by`).
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.server.served_id,
                            "object": "model",
                            "owned_by": "vllm",
                            "max_model_len": self.server.max_model_len,
                        }
                    ],
                },
            )
        elif self.path == _METRICS_PATH:
            # The in-flight counters both load paths read: this box's own
            # replica probe scrapes them directly, and a PEER sees the same
            # numbers one hop out through that box's gateway `/status`.
            self._send_text(
                200,
                f"vllm:num_requests_running {float(self.server.running)}\n"
                f"vllm:num_requests_waiting {float(self.server.waiting)}\n",
            )
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length > 0 else b""
        self._record(body)
        try:
            data = json.loads(body)
        except ValueError:
            data = {}
        if isinstance(data, dict) and data.get("stream") is True:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(_SSE_EVENTS[0])
            self.wfile.flush()
            gate = self.server.sse_gate
            if gate is not None:
                # Hold the SECOND event until the test's client confirms it
                # received the first — the deterministic incrementality proof.
                self.server.sse_gate_released = gate.wait(timeout=10)
            for event in _SSE_EVENTS[1:]:
                self.wfile.write(event)
                self.wfile.flush()
        elif self.server.post_status is not None:
            # The engine's own authoritative refusal (e.g. its 429 shed) —
            # relayed by the gateway exactly like any other 4xx.
            self._send_json(
                self.server.post_status, {"error": {"type": "server_busy", "message": "busy"}}
            )
        else:
            self._send_json(200, _chat_completion_body(self.server.served_id))


class _FakeBackend(ThreadingHTTPServer):
    """One fake vLLM engine on an ephemeral loopback port.

    ``running``/``waiting``/``post_status`` are plain mutable attributes a test
    sets between requests: they are what make "this replica is loaded" and
    "this replica refuses" controllable without a clock or a real engine.
    """

    def __init__(self, served_id: str, *, max_model_len: int = 262144) -> None:
        self.served_id = served_id
        self.max_model_len = max_model_len
        self.running = 0
        self.waiting = 0
        self.post_status: int | None = None
        self.log: list = []
        self.sse_gate: threading.Event | None = None
        self.sse_gate_released: bool | None = None
        super().__init__(("127.0.0.1", 0), _FakeBackendHandler)
        _serve_in_thread(self)
        host, port = self.server_address
        self.base = f"http://{host}:{port}"


# --- gateway spawn helpers ----------------------------------------------------


def _recording_handler(base_handler: type, log: list) -> type:
    """Wrap a bound gateway handler so every inbound request (method, path,
    headers, body) lands in ``log`` before normal processing — the capture
    point for "what did this box actually receive"."""

    class _Recording(base_handler):
        def do_GET(self) -> None:  # noqa: N802
            log.append(
                SimpleNamespace(
                    method="GET", path=self.path, headers=list(self.headers.items()), body=b""
                )
            )
            super().do_GET()

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length > 0 else b""
            log.append(
                SimpleNamespace(
                    method="POST", path=self.path, headers=list(self.headers.items()), body=body
                )
            )
            self.rfile = io.BytesIO(body)  # let the real handler re-read it
            super().do_POST()

    return _Recording


def _spawn_gateway(
    env: dict[str, str], *, log: list | None = None, pressure=None
) -> SimpleNamespace:
    """A REAL gateway: build_config → peer specs → a real ReadinessCache
    (``start=False``; tests seed it deterministically via ``refresh()``) → the
    real handler dispatch on a real ``ThreadingHTTPServer``. Nothing is
    monkeypatched — ``open_upstream`` opens genuine sockets."""
    table, cfg = build_config(env)
    specs = S.peer_specs_from_table(table, env)
    cache = ReadinessCache.from_backends(
        table.backends, peer_specs=tuple(specs.values()), start=False
    )
    handler = S._make_handler(table, cfg, pressure, cache, specs)
    if log is not None:
        handler = _recording_handler(handler, log)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    _serve_in_thread(httpd)
    host, port = httpd.server_address
    return SimpleNamespace(
        httpd=httpd, base=f"http://{host}:{port}", cache=cache, table=table, cfg=cfg, specs=specs
    )


def _shutdown(*servers) -> None:
    for srv in servers:
        srv.shutdown()
        srv.server_close()


@contextlib.contextmanager
def _two_gateways(peer_env_extra: dict[str, str] | None = None):
    """The two-box world: PEER gateway (senses hosted, own inbound key) and
    PROXYING gateway (spark-lobe shape, senses dropped+proxied to the peer)."""
    senses_backend = _FakeBackend(_SENSES_ID)
    peer_env = {
        # thor-lobe-shaped: cortex dropped (wired but infeasible + unreachable),
        # senses hosted at the fake engine, pairwise inbound auth armed.
        "PRIMARY_URL": _CLOSED_URL,
        "PRIMARY_SERVED_NAME": _CORTEX_ID,
        "PRIMARY_FEASIBLE": "false",
        "MULTIMODAL_BASE_URL": senses_backend.base,
        "MULTIMODAL_SERVED_NAME": _SENSES_ID,
        "GATEWAY_API_KEY": _PEER_KEY,
    }
    peer_env.update(peer_env_extra or {})
    peer_log: list = []
    peer = _spawn_gateway(peer_env, log=peer_log)
    peer.cache.refresh()  # deterministic seed: senses backend answers /health

    local_backend = _FakeBackend(_CORTEX_ID)
    spark_env = {
        # spark-lobe-shaped: cortex + pooling hosted locally, senses DROPPED
        # and proxied to the live peer with the pairwise key; own inbound key.
        "PRIMARY_URL": local_backend.base,
        "PRIMARY_SERVED_NAME": _CORTEX_ID,
        "EMBED_URL": local_backend.base,
        "EMBED_SERVED_NAME": _EMBED_ID,
        "RERANK_URL": local_backend.base,
        "RERANK_SERVED_NAME": _RERANK_ID,
        "MULTIMODAL_SERVED_NAME": _SENSES_ID,
        "MULTIMODAL_FEASIBLE": "false",
        "MULTIMODAL_PEER_ORIGIN": peer.base,
        "MULTIMODAL_PEER_PROXY": "true",
        "MULTIMODAL_PEER_API_KEY": _PEER_KEY,
        "GATEWAY_API_KEY": _CALLER_KEY,
    }
    box = _spawn_gateway(spark_env)
    box.cache.refresh()  # probes the local backends AND the live peer

    world = SimpleNamespace(
        box=box,
        peer=peer,
        peer_log=peer_log,
        senses_backend=senses_backend,
        local_backend=local_backend,
    )
    try:
        yield world
    finally:
        _shutdown(box.httpd, peer.httpd, senses_backend, local_backend)


@pytest.fixture
def world():
    with _two_gateways() as w:
        yield w


@pytest.fixture
def declining_world():
    """The misdeclared-referral shape: the PEER also dropped senses (wired but
    ``MULTIMODAL_FEASIBLE=false``), so it answers 404 ``role_infeasible``."""
    with _two_gateways({"MULTIMODAL_FEASIBLE": "false"}) as w:
        yield w


# --- client helpers -----------------------------------------------------------


def _request(base, path, *, method="GET", body=None, headers=None, key=_CALLER_KEY):
    all_headers = {"Content-Type": "application/json"} if body is not None else {}
    if key is not None:
        all_headers["Authorization"] = f"Bearer {key}"
    all_headers.update(headers or {})
    req = urllib.request.Request(base + path, data=body, method=method, headers=all_headers)
    return urllib.request.urlopen(req, timeout=10)


def _expect_error(code, base, path, *, method="GET", body=None, headers=None, key=_CALLER_KEY):
    with pytest.raises(urllib.error.HTTPError) as exc:
        _request(base, path, method=method, body=body, headers=headers, key=key)
    assert exc.value.code == code
    return exc.value


def _chat_body(model, **extra) -> bytes:
    payload = {"model": model, "messages": [{"role": "user", "content": "hi"}]}
    payload.update(extra)
    return json.dumps(payload).encode()


def _stop_peer(world) -> None:
    _shutdown(world.peer.httpd)


# ============================================================================
# (a) the proxied happy path: role name and tier alias, end to end
# ============================================================================


@pytest.mark.parametrize("alias", ["senses", "multimodal"])
def test_chat_naming_proxied_role_answers_from_peer_end_to_end(world, alias) -> None:
    with _request(
        world.box.base, "/v1/chat/completions", method="POST", body=_chat_body(alias)
    ) as resp:
        assert resp.status == 200
        # The declared origin, verbatim — the exact MULTIMODAL_PEER_ORIGIN value.
        assert resp.headers.get(S.PROXIED_BY_HEADER) == world.peer.base
        answer = json.loads(resp.read())
    # The body is the PEER's fake backend's canned completion, model id = the
    # peer's served id — never a locally-fabricated answer.
    assert answer == _chat_completion_body(_SENSES_ID)
    assert answer["model"] == _SENSES_ID
    # The request really crossed both hops: the peer gateway received exactly
    # one forwarded POST, and its own senses engine produced the answer.
    posts = [r for r in world.peer_log if r.method == "POST"]
    assert len(posts) == 1
    assert posts[0].path == "/v1/chat/completions"
    assert json.loads(posts[0].body)["model"] == _SENSES_ID
    assert any(r.method == "POST" for r in world.senses_backend.log)


def test_locally_served_role_answers_locally_without_proxied_marker(world) -> None:
    # Control for (a): a hosted role never touches the peer and never carries
    # the proxied-by marker.
    before = len(world.peer_log)
    with _request(
        world.box.base, "/v1/chat/completions", method="POST", body=_chat_body("cortex")
    ) as resp:
        assert resp.status == 200
        assert resp.headers.get(S.PROXIED_BY_HEADER) is None
        assert json.loads(resp.read())["model"] == _CORTEX_ID
    assert len(world.peer_log) == before


# ============================================================================
# (b) inbound auth on the proxying box: 401, and the peer never saw a request
# ============================================================================


@pytest.mark.parametrize("key", [None, "sk-wrong-key-entirely"])
def test_missing_or_wrong_key_401_and_peer_never_dialed(world, key) -> None:
    before = len(world.peer_log)
    err = _expect_error(
        401,
        world.box.base,
        "/v1/chat/completions",
        method="POST",
        body=_chat_body("senses"),
        key=key,
    )
    payload = json.loads(err.read())
    assert payload["error"]["code"] == "invalid_api_key"
    assert err.headers.get("WWW-Authenticate") == "Bearer"
    # The peer's inbound log is unchanged: the rejected request cost the mesh
    # zero cross-box sockets (and the local engine none either).
    assert len(world.peer_log) == before
    assert all(r.method == "GET" for r in world.senses_backend.log)  # health probes only


# ============================================================================
# (c) capabilities + /v1/models follow the LIVE peer, honestly
# ============================================================================


def test_capabilities_and_models_track_peer_lifecycle(world) -> None:
    # Peer up (seeded by the fixture's refresh): senses is proxied+ready and
    # its id is advertised.
    with _request(world.box.base, "/capabilities", key=None) as resp:
        caps = json.loads(resp.read())
    senses = caps["senses"]
    assert senses["proxied"] is True
    assert senses["hosted_by"] == world.peer.base  # the declared origin verbatim
    assert senses["ready"] is True  # the live peer probe verified /v1/models
    assert senses["feasible"] is False  # still a hardware fact, never relaxed
    assert "proxied" not in caps["cortex"]
    assert "hosted_by" not in caps["cortex"]
    with _request(world.box.base, "/v1/models") as resp:
        ids = {m["id"] for m in json.loads(resp.read())["data"]}
    assert ids == {_CORTEX_ID, _EMBED_ID, _RERANK_ID, _SENSES_ID}

    # Peer down + refresh: ready honest, the proxied id GONE from /v1/models.
    _stop_peer(world)
    world.box.cache.refresh()
    with _request(world.box.base, "/capabilities", key=None) as resp:
        caps = json.loads(resp.read())
    assert caps["senses"]["ready"] is False  # honest — never a hardcoded claim
    assert caps["senses"]["proxied"] is True  # config facts stay declared
    assert caps["senses"]["hosted_by"] == world.peer.base
    with _request(world.box.base, "/v1/models") as resp:
        ids = {m["id"] for m in json.loads(resp.read())["data"]}
    assert _SENSES_ID not in ids
    assert ids == {_CORTEX_ID, _EMBED_ID, _RERANK_ID}


# ============================================================================
# (d) loop guard: a marked request that would depart again is refused
# ============================================================================


def test_marked_request_refused_508_proxy_loop_peer_untouched(world) -> None:
    before = len(world.peer_log)
    err = _expect_error(
        508,
        world.box.base,
        "/v1/chat/completions",
        method="POST",
        body=_chat_body("senses"),
        headers={S.PROXIED_HEADER: "primary"},  # already crossed one hop elsewhere
    )
    payload = json.loads(err.read())
    assert payload["error"]["type"] == "proxy_loop"
    assert payload["error"]["code"] == "proxy_loop"
    assert payload["error"]["hops"] == ["primary", world.peer.base]
    assert err.headers.get(S.PROXIED_BY_HEADER) is None  # nothing was proxied
    assert len(world.peer_log) == before  # zero outbound attempts


# ============================================================================
# (e) peer down on the data plane: retryable 503 + Retry-After
# ============================================================================


def test_peer_down_chat_yields_503_with_retry_after(world) -> None:
    _stop_peer(world)
    err = _expect_error(
        503, world.box.base, "/v1/chat/completions", method="POST", body=_chat_body("senses")
    )
    assert err.headers.get("Retry-After") == str(S.BACKEND_UNAVAILABLE_RETRY_AFTER_SECONDS)
    assert err.headers.get(S.PROXIED_BY_HEADER) == world.peer.base  # names the failed peer
    payload = json.loads(err.read())
    assert payload["error"]["type"] == "backend_unavailable"
    assert world.peer.base in payload["error"]["message"]


# ============================================================================
# (f) peer declines role_infeasible: terminal, names the peer, one attempt
# ============================================================================


def test_peer_declining_role_infeasible_is_terminal_and_names_peer(declining_world) -> None:
    world = declining_world
    err = _expect_error(
        404, world.box.base, "/v1/chat/completions", method="POST", body=_chat_body("senses")
    )
    assert err.headers.get(S.PROXIED_BY_HEADER) == world.peer.base
    payload = json.loads(err.read())
    assert payload["error"]["type"] == "role_infeasible"
    assert payload["error"]["code"] == "role_infeasible"
    # The relayed error makes unmistakable that the DECLARED PEER declined —
    # a misdeclared referral — and keeps the peer's own verdict.
    assert world.peer.base in payload["error"]["message"]
    assert "declined" in payload["error"]["message"]
    assert "Peer said:" in payload["error"]["message"]
    # Exactly one outbound attempt — never a second hop.
    assert len([r for r in world.peer_log if r.method == "POST"]) == 1
    # The declining peer's engine was never dialed (its gateway 404'd first).
    assert all(r.method == "GET" for r in world.senses_backend.log)


# ============================================================================
# credential hygiene, end to end
# ============================================================================


def test_credential_hygiene_end_to_end(world, capfd) -> None:
    """The pairwise key is the Bearer on EVERY request the peer received; the
    caller's key reaches the peer in NONE of them; and neither key appears in
    anything the client can see (bodies, headers) or in the gateways' logs."""
    client_artifacts: list[str] = []

    def _see(status, headers, body: bytes) -> None:
        client_artifacts.append(f"{status}\n{headers}\n{body.decode('utf-8', 'replace')}")

    # A representative sweep: proxied chat, model listing, capabilities, the
    # loop refusal, a 401, and (after stopping the peer) the peer-down 503.
    with _request(
        world.box.base, "/v1/chat/completions", method="POST", body=_chat_body("senses")
    ) as r:
        _see(r.status, str(r.headers), r.read())
    with _request(world.box.base, "/v1/models") as r:
        _see(r.status, str(r.headers), r.read())
    with _request(world.box.base, "/capabilities", key=None) as r:
        _see(r.status, str(r.headers), r.read())
    err = _expect_error(
        508,
        world.box.base,
        "/v1/chat/completions",
        method="POST",
        body=_chat_body("senses"),
        headers={S.PROXIED_HEADER: "primary"},
    )
    _see(err.code, str(err.headers), err.read())
    err = _expect_error(
        401,
        world.box.base,
        "/v1/chat/completions",
        method="POST",
        body=_chat_body("senses"),
        key=None,
    )
    _see(err.code, str(err.headers), err.read())
    _stop_peer(world)
    world.box.cache.refresh()
    err = _expect_error(
        503, world.box.base, "/v1/chat/completions", method="POST", body=_chat_body("senses")
    )
    _see(err.code, str(err.headers), err.read())

    # The peer saw at least the readiness probes + the forwarded chat; on EVERY
    # one of them the ONLY credential is the pairwise key.
    assert len(world.peer_log) >= 2
    assert any(r.method == "POST" for r in world.peer_log)
    for received in world.peer_log:
        auth_values = [v for k, v in received.headers if k.lower() == "authorization"]
        assert auth_values == [f"Bearer {_PEER_KEY}"]
        dumped = json.dumps(received.headers) + received.body.decode("utf-8", "replace")
        assert _CALLER_KEY not in dumped
    # The caller's key was stripped BEFORE the box boundary, so it can never
    # reach the peer's own engine either; and the pairwise key never leaks
    # into the proxying box's LOCAL backend traffic.
    for received in world.senses_backend.log:
        assert _CALLER_KEY not in json.dumps(received.headers)
    for received in world.local_backend.log:
        assert _PEER_KEY not in json.dumps(received.headers)
    # Nothing the CLIENT saw — status lines, headers (markers included),
    # bodies (errors, capabilities) — contains either key string.
    for artifact in client_artifacts:
        assert _CALLER_KEY not in artifact
        assert _PEER_KEY not in artifact
    # And neither key reaches the gateways' captured log output.
    captured = capfd.readouterr()
    assert _CALLER_KEY not in captured.err
    assert _CALLER_KEY not in captured.out
    assert _PEER_KEY not in captured.err
    assert _PEER_KEY not in captured.out


# ============================================================================
# SSE end-to-end: the peer's stream arrives incrementally, marker present
# ============================================================================


def test_sse_stream_relays_incrementally_across_both_hops(world) -> None:
    gate = threading.Event()
    world.senses_backend.sse_gate = gate

    parts = urlsplit(world.box.base)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=10)
    try:
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=_chat_body("senses", stream=True),
            headers={
                "Authorization": f"Bearer {_CALLER_KEY}",
                "Content-Type": "application/json",
            },
        )
        resp = conn.getresponse()
        assert resp.status == 200
        assert resp.getheader(S.PROXIED_BY_HEADER) == world.peer.base
        # Read until the FIRST event has fully arrived — while the fake backend
        # is still holding the second one behind the gate.
        received = b""
        while _SSE_EVENTS[0] not in received:
            chunk = resp.read1(65536)
            assert chunk, "stream ended before the first SSE event arrived"
            received += chunk
        assert _SSE_EVENTS[1] not in received  # the gate really is still closed
        gate.set()  # release the rest of the stream
        while True:
            chunk = resp.read1(65536)
            if not chunk:
                break
            received += chunk
    finally:
        conn.close()
    # The gate was released BY THE CLIENT's receipt of event one — proof the
    # frames crossed both gateway hops incrementally, not in a terminal burst.
    assert world.senses_backend.sse_gate_released is True
    # And the byte stream is identical to what the peer's engine emitted.
    assert received == b"".join(_SSE_EVENTS)


# ============================================================================
# h7 goldens: a no-proxy/no-auth/no-peer deployment is byte-identical
# ============================================================================

# The oracle payloads, pinned as literal expected JSON. They are the exact
# PRE-FEATURE wire shapes: /capabilities carries no `proxied`/`hosted_by`, no
# response carries an auth challenge or a proxy marker, and the referral-only
# 404 (peer origin declared, proxy knob NOT armed — the issue #112 contract
# that predates #115/#127) is exactly the mesh-brain t3 body. Cross-checked
# against the shapes the pre-feature tests pin (tests/test_roles_proxied.py's
# oracle tests; tests/test_gateway_proxy.py's byte-identical referral-404
# test). Byte equality: the gateway renders `json.dumps(payload)` with default
# separators and insertion order, so `json.dumps(<literal>)` reproduces the
# wire bytes exactly.

_GOLDEN_HOST = "gateway.test:8000"
_GOLDEN_ORIGIN = "http://gateway.test:8000"
_REFERRAL_ORIGIN = "http://thor.local:8001"

_GOLDEN_CAPABILITIES = {
    "cortex": {
        "role": "cortex",
        "model": _CORTEX_ID,
        "runtime": "vllm",
        "endpoint": _GOLDEN_ORIGIN,
        "path": "/v1/chat/completions",
        "context": 262144,
        "quant": "modelopt",
        "mtp": True,
        "tools": True,
        "responsibilities": [
            "reasoning",
            "deciding",
            "planning",
            "tool_use",
            "code_repo_actions",
            "validation",
            "final_authority",
            # 2026-07-31: the cortex checkpoint became multimodal, so the advert
            # must say so — a consumer resolving roles by name (and NEVER parsing
            # model ids, as the contract instructs) would otherwise read a seeing
            # cortex as blind. Reported by colleague#361.
            "image_understanding",
            "video_understanding",
        ],
        "forbidden_responsibilities": [],
        "feasible": True,
        "ready": True,
        "loaded": True,
    },
    "senses": {
        "role": "senses",
        "model": _SENSES_ID,
        "runtime": "vllm",
        "endpoint": _GOLDEN_ORIGIN,
        "path": "/v1/chat/completions",
        "context": None,
        "quant": "compressed-tensors",
        "mtp": True,
        # senses serves `pythonic` tool calls (the 12B Gemma gear) — `tools` is
        # a CAPABILITY of the lane, independent of `tool_use` being absent from
        # its responsibilities (a division-of-labour statement, not a wiring one).
        "tools": True,
        "responsibilities": [
            "intake",
            "normalize_input",
            "classify_intent",
            "prepare_context_packet",
            "speak_back",
        ],
        "forbidden_responsibilities": [
            "final_decision",
            "repo_action",
            "security_decision",
        ],
        "feasible": False,
        "ready": False,
        "loaded": False,
    },
    "muse": {
        "role": "muse",
        # The opt-in muse lobe: unwired in this no-knob deployment, so it is
        # honestly infeasible-by-default (OPT_IN_BACKENDS) and named by its
        # catalog default. mtp True: the catalog entry DECLARES the assistant
        # MTP draft (unmeasured — see the entry's comment).
        "model": "nvidia/Gemma-4-31B-IT-NVFP4",
        "runtime": "vllm",
        "endpoint": _GOLDEN_ORIGIN,
        "path": "/v1/chat/completions",
        "context": None,
        "quant": "modelopt",
        "mtp": True,
        # True even though this deployment does not HOST muse: `tools` is a fact
        # about the model the role would serve (the catalog's `pythonic` parser),
        # exactly like `model`/`context`/`quant`/`mtp` above it. `feasible: false`
        # is what tells a caller it is unreachable here.
        "tools": True,
        "responsibilities": [
            "creative_generation",
            "long_form_writing",
            "ideation",
            "style_variation",
            "divergent_second_opinion",
            "tool_use",
        ],
        "forbidden_responsibilities": [
            "final_decision",
            "repo_action",
            "security_decision",
        ],
        "feasible": False,
        "ready": False,
        "loaded": False,
    },
    "worker": {
        "role": "worker",
        # The opt-in worker lobe: unwired in this no-knob deployment, so it is
        # honestly infeasible-by-default (OPT_IN_BACKENDS, exactly like muse
        # above) and named by its catalog default. Catalog worker moved to
        # nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
        # (nemotron-lightning-worker plan, #187, t3) — text-only, 1M native
        # ceiling, nvidia modelopt quant. mtp False: config.json carries no
        # MTP/draft-head field for this checkpoint (unlike the demoted
        # Qwen worker's self-hosted draft); the card's separate MTP/DSpark
        # claim is declared, unmeasured (plan t2).
        #
        # NOTE: `responsibilities` below is roles.py's OWN static vocabulary
        # (ROLE_RESPONSIBILITIES), not derived from the catalog — it still
        # names image_understanding/video_understanding here because the
        # sibling nemotron-lightning-worker plan task t4 (roles.py) redefines
        # that vocabulary for the new TEXT-ONLY checkpoint; this task (t3)
        # only changes the catalog entry.
        "model": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
        "runtime": "vllm",
        "endpoint": _GOLDEN_ORIGIN,
        "path": "/v1/chat/completions",
        "context": None,
        "quant": "modelopt",
        "mtp": False,
        # True even though this deployment does not HOST worker: `tools` is a
        # fact about the model the role would serve (the catalog's
        # `qwen3_coder` tool parser — UNVALIDATED on our engine, cited from
        # the checkpoint's own card, see lobes/catalog.py), exactly like muse
        # above. `feasible: false` is what tells a caller it is unreachable
        # here.
        "tools": True,
        "responsibilities": [
            "execution",
            "ground_work",
            "bulk_transform",
            "drafting",
            "action_selection",
            "retrieval_synthesis",
            "summarization",
            "log_digestion",
            "structured_extraction",
            "repo_inspection",
            "run_authorized_commands",
            "tool_use",
            "repo_action",
        ],
        # Unlike muse/senses, worker MAY act on the repo — repo_action is
        # deliberately ABSENT here (it is only permitted, never forbidden).
        # code_authoring IS forbidden (issue #187): "not coder" does not mean
        # "cannot touch a repository" — worker may inspect/run, never author.
        "forbidden_responsibilities": [
            "final_decision",
            "security_decision",
            "code_authoring",
        ],
        "feasible": False,
        "ready": False,
        "loaded": False,
    },
    "associate": {
        "role": "associate",
        # The TENTH Colleague role (lightning-on-orin plan, t6): worker MINUS
        # repo_action. Opt-in like muse/worker, unwired in this no-knob
        # deployment, so honestly infeasible-by-default (OPT_IN_BACKENDS) and
        # named by the catalog gear it shares with `worker` — one checkpoint,
        # two public addresses with different authority.
        "model": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
        "runtime": "vllm",
        "endpoint": _GOLDEN_ORIGIN,
        "path": "/v1/chat/completions",
        "context": None,
        "quant": "modelopt",
        "mtp": False,
        "tools": True,
        "responsibilities": [
            "execution",
            "ground_work",
            "bulk_transform",
            "drafting",
            "repo_inspection",
            "run_authorized_commands",
            "tool_use",
        ],
        # The ONE token that separates associate from worker: repo_action is
        # FORBIDDEN here and merely absent-from-forbidden there. "They do, but
        # not act."
        "forbidden_responsibilities": [
            "final_decision",
            "security_decision",
            "code_authoring",
            "repo_action",
        ],
        "feasible": False,
        "ready": False,
        "loaded": False,
    },
    "hand": {
        "role": "hand",
        # The ninth Colleague role. DEFAULT-HOSTED, so unlike muse/worker above
        # it is NOT in OPT_IN_BACKENDS — a no-knob deployment that has not been
        # re-inited reads it as the SLEEPING LOBE: feasible true (this card can
        # obviously serve 2.4 GiB of bf16), ready/loaded false (the lane is not
        # actually up). Advertising it ready would be the #92 defect; declaring
        # it infeasible would be the opposite lie.
        "model": "LiquidAI/LFM2.5-1.2B-Instruct",
        "runtime": "vllm",
        "endpoint": _GOLDEN_ORIGIN,
        "path": "/v1/chat/completions",
        "context": 32768,
        # bf16 — the catalog's "none" sentinel, meaning the lane omits
        # --quantization entirely rather than passing it empty.
        "quant": "none",
        "mtp": False,
        "tools": True,
        "responsibilities": [
            "domain_mastery",
            "learned_skill",
            "specialized_task",
            "tool_use",
        ],
        # v1 withholds repo_action: adding a responsibility later is
        # contract-compatible, removing one is a break (issue #180).
        "forbidden_responsibilities": [
            "final_decision",
            "repo_action",
            "security_decision",
        ],
        "feasible": True,
        "ready": False,
        "loaded": False,
    },
    "embedder": {
        "role": "embedder",
        "model": _EMBED_ID,
        "runtime": "vllm",
        "endpoint": _GOLDEN_ORIGIN,
        "path": "/v1/embeddings",
        "context": 32768,
        "quant": "",
        "mtp": False,
        "tools": False,  # pooling lane — no chat endpoint to accept `tools`
        "responsibilities": ["vectorization", "memory_retrieval_input"],
        "forbidden_responsibilities": [],
        "feasible": True,
        "ready": False,
        "loaded": False,
    },
    "reranker": {
        "role": "reranker",
        "model": _RERANK_ID,
        "runtime": "vllm",
        "endpoint": _GOLDEN_ORIGIN,
        "path": "/v1/rerank",
        "context": 32768,
        "quant": "",
        "mtp": False,
        "tools": False,  # pooling lane — no chat endpoint to accept `tools`
        "responsibilities": ["retrieval_ordering", "relevance_refinement"],
        "forbidden_responsibilities": [],
        "feasible": True,
        "ready": False,
        "loaded": False,
    },
    "stt": {
        "role": "stt",
        "model": "nvidia/parakeet-tdt-0.6b-v2",
        "runtime": "parakeet",
        "endpoint": "",
        "path": "/v1/audio/transcriptions",
        "context": 0,
        "quant": "",
        "mtp": False,
        "tools": False,  # audio sidecar — transcription, not a chat lane
        "responsibilities": ["transcribe", "audio_input_to_text"],
        "forbidden_responsibilities": [],
        "feasible": True,
        "ready": False,
        "loaded": False,
    },
    "tts": {
        "role": "tts",
        "model": "ResembleAI/chatterbox",
        "runtime": "chatterbox",
        "endpoint": "",
        "path": "/v1/audio/speech",
        "context": 0,
        "quant": "",
        "mtp": False,
        "tools": False,  # audio sidecar — synthesis, not a chat lane
        "responsibilities": ["speech_output", "synthesize"],
        "forbidden_responsibilities": [],
        "feasible": True,
        "ready": False,
        "loaded": False,
    },
}

_GOLDEN_MODELS = {
    "object": "list",
    "data": [{"id": _CORTEX_ID, "object": "model", "owned_by": "lobes"}],
}

_INFEASIBLE_MESSAGE = (
    "The model `senses` is not feasible on this machine — its backend "
    "(`multimodal`) is declared hardware-infeasible by this deployment's "
    "per-machine profile and will never be served here."
)

_GOLDEN_404_NO_PEER = {
    "error": {
        "message": _INFEASIBLE_MESSAGE,
        "type": "role_infeasible",
        "code": "role_infeasible",
    }
}

_GOLDEN_404_REFERRAL_ONLY = {
    "error": {
        "message": (
            _INFEASIBLE_MESSAGE
            + f" It is hosted by the peer at `{_REFERRAL_ORIGIN}` — address that "
            "box directly; this gateway never proxies requests to peers."
        ),
        "type": "role_infeasible",
        "code": "role_infeasible",
        "hosted_by": _REFERRAL_ORIGIN,
    }
}


def _assert_no_feature_trace(headers, body: bytes) -> None:
    """No wire trace of proxy-lobes/pairwise-auth: no proxy markers, no auth
    challenge, no proxied capability key."""
    assert headers.get(S.PROXIED_BY_HEADER) is None
    assert headers.get(S.PROXIED_HEADER) is None
    assert headers.get("WWW-Authenticate") is None
    assert b'"proxied"' not in body


@pytest.fixture
def golden_gateway(monkeypatch):
    """A gateway from the minimal no-knob env (no proxy, no auth, no peer):
    a hosted primary + the pre-feature dropped-senses channel (#113) only."""
    # capabilities_payload reads the served-context overlay from os.environ on
    # the HTTP route — scrub it so the golden bytes are deterministic.
    for var in (
        "PRIMARY_MAX_MODEL_LEN",
        "MULTIMODAL_MAX_MODEL_LEN",
        "EMBED_MAX_MODEL_LEN",
        "RERANK_MAX_MODEL_LEN",
    ):
        monkeypatch.delenv(var, raising=False)
    backend = _FakeBackend(_CORTEX_ID)
    env = {
        "PRIMARY_URL": backend.base,
        "PRIMARY_SERVED_NAME": _CORTEX_ID,
        "MULTIMODAL_FEASIBLE": "false",
    }
    gw = _spawn_gateway(env)
    gw.cache.refresh()
    try:
        yield gw
    finally:
        _shutdown(gw.httpd, backend)


def test_golden_capabilities_bytes_no_knob_deployment(golden_gateway) -> None:
    # Explicit Host so the echoed origin (issue #87) is deterministic.
    with _request(
        golden_gateway.base, "/capabilities", key=None, headers={"Host": _GOLDEN_HOST}
    ) as resp:
        raw = resp.read()
        assert raw == json.dumps(_GOLDEN_CAPABILITIES).encode("utf-8")
        _assert_no_feature_trace(resp.headers, raw)
    assert b"hosted_by" not in raw


def test_golden_v1_models_bytes_no_knob_deployment(golden_gateway) -> None:
    with _request(golden_gateway.base, "/v1/models", key=None) as resp:
        raw = resp.read()
        assert raw == json.dumps(_GOLDEN_MODELS).encode("utf-8")
        _assert_no_feature_trace(resp.headers, raw)


def test_golden_role_infeasible_404_bytes_no_peer_config(golden_gateway) -> None:
    err = _expect_error(
        404,
        golden_gateway.base,
        "/v1/chat/completions",
        method="POST",
        body=_chat_body("senses"),
        key=None,
    )
    raw = err.read()
    assert raw == json.dumps(_GOLDEN_404_NO_PEER).encode("utf-8")
    _assert_no_feature_trace(err.headers, raw)
    assert b"hosted_by" not in raw


def test_golden_role_infeasible_404_bytes_referral_only() -> None:
    # The referral-only shape (origin declared, proxy knob NOT armed) is the
    # PRE-#127 issue #112 contract — its 404 must also stay byte-identical.
    backend = _FakeBackend(_CORTEX_ID)
    env = {
        "PRIMARY_URL": backend.base,
        "PRIMARY_SERVED_NAME": _CORTEX_ID,
        "MULTIMODAL_FEASIBLE": "false",
        "MULTIMODAL_PEER_ORIGIN": _REFERRAL_ORIGIN,
    }
    gw = _spawn_gateway(env)
    try:
        gw.cache.refresh()
        err = _expect_error(
            404,
            gw.base,
            "/v1/chat/completions",
            method="POST",
            body=_chat_body("senses"),
            key=None,
        )
        raw = err.read()
        assert raw == json.dumps(_GOLDEN_404_REFERRAL_ONLY).encode("utf-8")
        _assert_no_feature_trace(err.headers, raw)
    finally:
        _shutdown(gw.httpd, backend)


# ============================================================================
# The replica pool, end to end: N REAL gateways on loopback (t9, issue #199)
# ============================================================================
#
# Everything above proves the PROXY branch (a role this box does not host).
# The pool is the other direction: a role every box DOES host, placed per
# request onto whichever replica the live snapshot says is least loaded. The
# per-task suites (tests/test_gateway_pool.py,
# tests/test_gateway_pool_pressure.py, tests/test_replicas.py) drive that
# through injected seams — a fake `replica_snapshot`, a scripted opener, a
# fixed pressure dict. Nothing there ever runs a REAL probe against a REAL
# peer gateway, so nothing there can catch a fingerprint that fails to
# round-trip through `/capabilities`, a load number that never reaches
# `/status`, or a marker that a live relay drops.
#
# `_n_gateways` closes that gap: N gateways, each with its own fake vLLM
# engine, each declaring the others as `PRIMARY_PEER_ORIGINS` replicas of the
# same `cortex`. Every hop is a genuine socket. Determinism comes from
# refreshing the caches SYNCHRONOUSLY (`start=False` — the same discipline the
# ReadinessCache harness above uses) and from a hand-driven pressure provider,
# never from sleeping past a background interval.

_POOL_CORTEX_ID = "unsloth/Qwen3.8-27B-NVFP4"
_POOL_MAX_MODEL_LEN = 262144
_POOL_QUANTIZATION = "compressed-tensors"

_POOL_NO_PRESSURE = {"swap_used_percent": 0.0, "iowait_percent": 0.0}
_POOL_HIGH_PRESSURE = {"swap_used_percent": 90.0, "iowait_percent": 90.0}
# Deviation d1: the two host signals split — swap is verifiable thrash and
# still sheds; iowait alone no longer refuses POOLED work.
_POOL_SWAP_ONLY_PRESSURE = {"swap_used_percent": 90.0, "iowait_percent": 0.0}
_POOL_IOWAIT_ONLY_PRESSURE = {"swap_used_percent": 0.0, "iowait_percent": 90.0}

_READY_TIMEOUT_SECONDS = 10.0


class _ManualPressure:
    """A hand-driven stand-in for :class:`~lobes.gateway._tier_request.PressureCache`.

    The handler only ever calls ``current()``, so a two-line duck type keeps
    the busy/idle verdict a TEST DECISION rather than a race against a 2 s
    background sampler. Flipping it is instantaneous and total-ordered with
    the request that reads it.
    """

    def __init__(self) -> None:
        self.value: dict[str, float] = dict(_POOL_NO_PRESSURE)

    def current(self) -> dict[str, float]:
        return dict(self.value)

    def set_busy(self, busy: bool = True) -> None:
        self.value = dict(_POOL_HIGH_PRESSURE if busy else _POOL_NO_PRESSURE)


def _reserve_gateway() -> tuple[ThreadingHTTPServer, str]:
    """Bind a loopback port WITHOUT serving, so its origin is known before the
    handler exists.

    The pool is circular by construction — box *i*'s env names every other
    box's origin — so the ports must all be known before any table is built.
    Binding first and swapping ``RequestHandlerClass`` in afterwards keeps
    that race-free; reserving a port with a throwaway socket and re-binding it
    would not.
    """
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    host, port = httpd.server_address
    return httpd, f"http://{host}:{port}"


def _pool_env(backend_base: str, self_origin: str, peers: Sequence[str]) -> dict[str, str]:
    """One box's `.env`: hosts cortex locally, declares every other box as a
    replica of the SAME role, with an empty (no-inbound-gate) key slot each."""
    return {
        "PRIMARY_URL": backend_base,
        "PRIMARY_SERVED_NAME": _POOL_CORTEX_ID,
        # The declared half of the fingerprint. Without it the lane's
        # quantization reads `unknown`, and the unknown-rule (spec h11) would
        # disqualify every peer — the pool would silently never form.
        "PRIMARY_QUANTIZATION": _POOL_QUANTIZATION,
        "GATEWAY_SELF_ORIGIN": self_origin,
        "PRIMARY_PEER_ORIGINS": ",".join(peers),
        "PRIMARY_PEER_API_KEYS": ",".join("" for _ in peers),
    }


def _bind_handler(box) -> None:
    """(Re)bind this box's handler class from its current caches, live."""
    box.httpd.RequestHandlerClass = _recording_handler(
        S._make_handler(
            box.table,
            box.cfg,
            box.pressure,
            box.cache,
            box.specs,
            S.replica_snapshot_provider(box.replicas),
            box.replicas,
        ),
        box.log,
    )


def _wire_gateway(httpd, base: str, env: dict[str, str], log: list):
    """Serve an already-bound gateway with the pool DORMANT (no caches yet).

    Two phases, deliberately: :func:`S.build_replica_caches` probes every
    declared peer synchronously, and a peer that is bound-but-not-yet-serving
    accepts the connection and then says nothing — so building the caches
    before every box is answering would stall each box for the full 3 s probe
    timeout. Serving first, then attaching the caches (:func:`_attach_replicas`)
    keeps the harness fast AND keeps the probes real.
    """
    table, cfg = build_config(env)
    specs = S.peer_specs_from_table(table, env)
    box = SimpleNamespace(
        httpd=httpd,
        base=base,
        table=table,
        cfg=cfg,
        specs=specs,
        cache=ReadinessCache.from_backends(
            table.backends, peer_specs=tuple(specs.values()), start=False
        ),
        replicas={},
        pressure=_ManualPressure(),
        log=log,
    )
    _bind_handler(box)
    _serve_in_thread(httpd)
    return box


def _attach_replicas(box) -> None:
    """Build this box's live ReplicaCaches and rebind the handler onto them.

    ``start=False``: no daemon threads, so every snapshot a test reads is one
    a test explicitly asked for. The constructor still refreshes once, which
    populates the LOCAL fingerprint (its engine is already up) — that is what
    a peer reads off this box's ``/capabilities``.
    """
    box.replicas = S.build_replica_caches(box.table, start=False)
    _bind_handler(box)


def _posts(box) -> list:
    """Every POST this gateway actually received (probe GETs excluded)."""
    return [r for r in box.log if r.method == "POST"]


class _Pool:
    """The N-box world, plus the two verbs every scenario needs."""

    def __init__(self, boxes, backends) -> None:
        self.boxes = list(boxes)
        self.backends = list(backends)

    def __getitem__(self, index: int):
        return self.boxes[index]

    def refresh(self) -> None:
        """One synchronous probe pass on every box, local lanes before peers.

        Two passes over the replica caches, not one: a peer's compatibility is
        decided against the fingerprint it publishes on ITS /capabilities,
        which is only as fresh as ITS OWN last local probe. Refreshing every
        local lane first makes the peer pass read current fingerprints rather
        than whatever the previous pass happened to leave behind.
        """
        for box in self.boxes:
            box.cache.refresh()
            for cache in box.replicas.values():
                cache._refresh_local()  # noqa: SLF001 - the local half only
        for box in self.boxes:
            for cache in box.replicas.values():
                cache.refresh()

    def wait_ready(self, box_index: int = 0, *, expect: bool = True) -> list[dict]:
        """Refresh until this box's own /capabilities agrees about its peers.

        Bounded polling on the OBSERVABLE surface — never a fixed sleep — so a
        slow loopback round trip costs latency, not a flake.
        """
        deadline = _monotonic() + _READY_TIMEOUT_SECONDS
        rows: list[dict] = []
        while True:
            self.refresh()
            rows = _capabilities_replicas(self.boxes[box_index])
            peers = [row for row in rows if not row["local"]]
            if peers and all(row["ready"] is expect for row in peers):
                return rows
            if _monotonic() > deadline:  # pragma: no cover - only on a wedged box
                raise AssertionError(f"replicas never reached ready={expect}: {rows}")


def _capabilities_replicas(box) -> list[dict]:
    with _request(box.base, "/capabilities", key=None) as resp:
        payload = json.loads(resp.read())
    return payload["cortex"]["replicas"]


@contextlib.contextmanager
def _n_gateways(n: int = 2, pool_env=None):
    """N real loopback gateways, each a replica of the same ``cortex``.

    ``pool_env`` is either one dict applied to every box or a per-box sequence
    of dicts (``None`` for "no override"), so a scenario can arm an inbound
    key on one box and the matching outbound slot on another.
    """
    backends = [_FakeBackend(_POOL_CORTEX_ID, max_model_len=_POOL_MAX_MODEL_LEN) for _ in range(n)]
    reserved = [_reserve_gateway() for _ in range(n)]
    origins = [origin for _, origin in reserved]
    if pool_env is None or isinstance(pool_env, Mapping):
        overrides = [dict(pool_env or {}) for _ in range(n)]
    else:
        overrides = [dict(item or {}) for item in pool_env]
    boxes = []
    try:
        for index, (httpd, base) in enumerate(reserved):
            env = _pool_env(
                backends[index].base,
                origins[index],
                [o for j, o in enumerate(origins) if j != index],
            )
            env.update(overrides[index])
            boxes.append(_wire_gateway(httpd, base, env, []))
        for box in boxes:
            _attach_replicas(box)
        pool = _Pool(boxes, backends)
        pool.refresh()
        yield pool
    finally:
        _shutdown(*[b.httpd for b in boxes], *backends)


@pytest.fixture
def pool():
    with _n_gateways(2) as p:
        p.wait_ready()
        yield p


def _pool_chat(box, model: str = "cortex", *, headers=None, key=None):
    return _request(
        box.base, _CHAT_PATH, method="POST", body=_chat_body(model), headers=headers, key=key
    )


def _serving_origin(resp) -> str:
    """Who answered: the forwarding attribution when relayed, else this box.

    ``X-Lobes-Proxied-By`` wins deliberately — a relayed answer also carries
    the PEER's own ``X-Lobes-Served-By`` (the peer stamped it before handing
    the bytes back), and the caller's question is "which box produced this",
    which the proxy attribution answers.
    """
    proxied = resp.headers.get(S.PROXIED_BY_HEADER)
    return proxied if proxied else resp.headers.get(S.SERVED_BY_HEADER)


# --- (1) spread: the pool actually places work off a loaded box --------------


def test_pool_forwards_to_the_idle_peer_when_this_box_is_loaded(pool) -> None:
    spark, thor = pool[0], pool[1]
    pool.backends[0].running, pool.backends[0].waiting = 3, 2
    pool.wait_ready()
    with _pool_chat(spark) as resp:
        assert resp.status == 200
        assert resp.headers.get(S.PROXIED_BY_HEADER) == thor.base
        assert resp.headers.get(S.ROUTE_REASON_HEADER) == REASON_PEER_LESS_LOADED
        assert json.loads(resp.read())["model"] == _POOL_CORTEX_ID
    # It really crossed the wire: the peer gateway received exactly one
    # forwarded POST and its OWN engine produced the answer.
    assert len(_posts(thor)) == 1
    assert any(r.method == "POST" for r in pool.backends[1].log)
    assert not any(r.method == "POST" for r in pool.backends[0].log)


def test_pool_serves_locally_when_both_replicas_are_idle(pool) -> None:
    spark, thor = pool[0], pool[1]
    with _pool_chat(spark) as resp:
        assert resp.status == 200
        assert resp.headers.get(S.SERVED_BY_HEADER) == spark.base
        assert resp.headers.get(S.ROUTE_REASON_HEADER) == REASON_LOCAL_IDLE
        assert resp.headers.get(S.PROXIED_BY_HEADER) is None
    assert _posts(thor) == []
    assert any(r.method == "POST" for r in pool.backends[0].log)


def test_concurrent_requests_to_a_loaded_box_reach_the_pool(pool) -> None:
    # The exact split is policy-dependent (and the snapshot is deliberately
    # frozen between refreshes), so this asserts only what the pool GUARANTEES:
    # every request is attributed to a replica, and the loaded box's peer is
    # among the boxes that served.
    spark, thor = pool[0], pool[1]
    pool.backends[0].running, pool.backends[0].waiting = 3, 2
    pool.wait_ready()
    origins: list[str] = []
    lock = threading.Lock()

    def one() -> None:
        with _pool_chat(spark) as resp:
            assert resp.status == 200
            origin = _serving_origin(resp)
        with lock:
            origins.append(origin)

    threads = [threading.Thread(target=one) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
    assert len(origins) == 3
    assert set(origins) <= {spark.base, thor.base}
    assert thor.base in origins


# --- (2) peer down ----------------------------------------------------------


def test_a_stopped_peer_leaves_the_local_replica_sole_ready(pool) -> None:
    spark, thor = pool[0], pool[1]
    pool.backends[0].running = 9  # loaded, so ONLY the peer's death keeps it local
    _shutdown(thor.httpd)
    rows = pool.wait_ready(expect=False)
    peer_row = next(row for row in rows if not row["local"])
    assert peer_row["origin"] == thor.base
    assert peer_row["ready"] is False
    with _pool_chat(spark) as resp:
        assert resp.status == 200
        assert resp.headers.get(S.SERVED_BY_HEADER) == spark.base
        assert resp.headers.get(S.ROUTE_REASON_HEADER) == REASON_SOLE_READY
        assert resp.headers.get(S.PROXIED_BY_HEADER) is None


# --- (3) local pressure forwards instead of shedding ------------------------


def test_local_pressure_forwards_a_pooled_request_to_the_peer(pool) -> None:
    spark, thor = pool[0], pool[1]
    spark.pressure.set_busy()
    with _pool_chat(spark) as resp:
        assert resp.status == 200
        assert resp.headers.get(S.PROXIED_BY_HEADER) == thor.base
        assert resp.headers.get(S.ROUTE_REASON_HEADER) == REASON_LOCAL_BUSY_FORWARDED
    assert len(_posts(thor)) == 1
    assert not any(r.method == "POST" for r in pool.backends[0].log)


def test_a_peers_refusal_is_relayed_after_exactly_one_forward(pool) -> None:
    # The one-forward rule (c35/h27): the peer's own authoritative refusal
    # rides back untouched — never retried locally (pressure forbade that) and
    # never forwarded onward, which is what would ping-pong two loaded boxes.
    #
    # The refusal is the PEER ENGINE's 429 rather than the peer gateway's own
    # pressure shed, and that is not a shortcut: a forwarded pooled request
    # arrives with its `model` already rewritten to the raw served id (see
    # `_relay_to_target`), which is not a tier alias, so the receiving
    # gateway's tier/pressure branch never runs for it. A peer under pressure
    # therefore cannot shed a forwarded pooled request at the tier gate at all
    # — the pool's own snapshot (`busy`) is what keeps work off a loaded peer,
    # and the test below covers that half.
    spark, thor = pool[0], pool[1]
    spark.pressure.set_busy()
    pool.backends[1].post_status = 429
    err = _expect_error(429, spark.base, _CHAT_PATH, method="POST", body=_chat_body("cortex"))
    assert err.headers.get(S.PROXIED_BY_HEADER) == thor.base
    assert len(_posts(thor)) == 1  # exactly one forward, never two
    assert not any(r.method == "POST" for r in pool.backends[0].log)


def test_both_boxes_busy_forwards_once_and_the_peer_serves(pool) -> None:
    # RENEGOTIATED under deviation d1 — was
    # `test_both_boxes_busy_sheds_locally_and_never_forwards`, which asserted
    # a local 429 with the `none` route reason and zero outbound sockets.
    #
    # Two things changed and they compose. t3 stopped `_is_selectable` keying
    # pool candidacy on a PEER's host pressure verdict, so the peer is a
    # candidate again — both engines here are genuinely idle. And d1 stopped
    # the RECEIVER shedding a pooled arrival on a host-level verdict alone, so
    # the forward is served rather than bounced back as a second 429.
    #
    # The one-forward rule is untouched: EXACTLY one outbound POST, and the
    # peer never forwards it onward.
    spark, thor = pool[0], pool[1]
    thor.pressure.set_busy()
    pool.refresh()  # the snapshot now carries the peer's own busy verdict
    spark.pressure.set_busy()
    with _pool_chat(spark) as resp:
        assert resp.status == 200
        assert resp.headers.get(S.PROXIED_BY_HEADER) == thor.base
        assert resp.headers.get(S.ROUTE_REASON_HEADER) == REASON_LOCAL_BUSY_FORWARDED
    assert len(_posts(thor)) == 1
    assert not any(r.method == "POST" for r in pool.backends[0].log)


def test_a_pooled_arrival_is_served_under_iowait_only_pressure(pool) -> None:
    # d1's receiving side, end to end over real sockets: the box's ONLY
    # complaint is a host iowait reading (swap at zero) and its engine is
    # idle, so the request it was forwarded is served here — no 429, no
    # second hop. This is the c17 success signal in miniature.
    spark, thor = pool[0], pool[1]
    thor.pressure.value = dict(_POOL_IOWAIT_ONLY_PRESSURE)
    before = len(pool.backends[0].log)
    with _pool_chat(thor, headers={S.PROXIED_HEADER: "primary"}) as resp:
        assert resp.status == 200
        assert resp.headers.get(S.SERVED_BY_HEADER) == thor.base
        assert resp.headers.get(S.ROUTE_REASON_HEADER) == REASON_SOLE_READY
    assert any(r.method == "POST" for r in pool.backends[1].log)
    assert len(pool.backends[0].log) == before
    assert _posts(spark) == []


def test_a_pooled_arrival_is_still_shed_under_swap_thrash(pool) -> None:
    # The line, over real sockets: same arrival, swap over the threshold and
    # iowait at zero. A genuinely paging box refuses, and still never
    # re-forwards.
    spark, thor = pool[0], pool[1]
    thor.pressure.value = dict(_POOL_SWAP_ONLY_PRESSURE)
    err = _expect_error(
        429,
        thor.base,
        _CHAT_PATH,
        method="POST",
        body=_chat_body("cortex"),
        headers={S.PROXIED_HEADER: "primary"},
    )
    assert err.headers.get("X-Lobes-Tier-Reason") == "busy"
    assert _posts(spark) == []


# --- (4) single hop ---------------------------------------------------------


def test_a_marked_arrival_is_served_locally_and_never_re_placed(pool) -> None:
    spark, thor = pool[0], pool[1]
    pool.backends[1].running, pool.backends[1].waiting = 5, 5  # thor is the LOADED box
    pool.wait_ready()
    before = len(pool.backends[0].log)
    with _pool_chat(thor, headers={S.PROXIED_HEADER: "primary"}) as resp:
        assert resp.status == 200
        assert resp.headers.get(S.SERVED_BY_HEADER) == thor.base
        assert resp.headers.get(S.ROUTE_REASON_HEADER) == REASON_SOLE_READY
        assert resp.headers.get(S.PROXIED_BY_HEADER) is None
    assert any(r.method == "POST" for r in pool.backends[1].log)
    # The other box's engine never saw it — the marked arrival was not placed.
    assert len(pool.backends[0].log) == before
    assert _posts(spark) == []


# --- (5) raw id and alias take the identical path ---------------------------


@pytest.mark.parametrize("loaded", [False, True])
def test_raw_served_id_and_alias_place_identically(pool, loaded) -> None:
    spark = pool[0]
    if loaded:
        pool.backends[0].running, pool.backends[0].waiting = 3, 2
    pool.wait_ready()
    with _pool_chat(spark, "cortex") as resp:
        by_alias = (_serving_origin(resp), resp.headers.get(S.ROUTE_REASON_HEADER))
    with _pool_chat(spark, _POOL_CORTEX_ID) as resp:
        by_raw_id = (_serving_origin(resp), resp.headers.get(S.ROUTE_REASON_HEADER))
    assert by_alias == by_raw_id
    assert by_alias[1] == (REASON_PEER_LESS_LOADED if loaded else REASON_LOCAL_IDLE)


# --- (6) affinity -----------------------------------------------------------


def _affinity_origin(box, key: str) -> str:
    with _pool_chat(box, headers={S.AFFINITY_HEADER: key}) as resp:
        assert resp.status == 200
        return _serving_origin(resp)


def test_one_affinity_key_sticks_to_one_replica(pool) -> None:
    spark = pool[0]
    seen = {_affinity_origin(spark, "sess-1") for _ in range(5)}
    assert len(seen) == 1
    assert seen <= {spark.base, pool[1].base}


def test_a_different_affinity_key_may_land_elsewhere(pool) -> None:
    # Rendezvous hashing gives no guarantee that two keys DIFFER — only that
    # each is stable. So this pins what is actually contracted: whatever a new
    # key resolves to is a real replica of the pool, and it is stable too.
    spark = pool[0]
    origins = {key: _affinity_origin(spark, key) for key in ("sess-1", "sess-2", "sess-3")}
    assert set(origins.values()) <= {spark.base, pool[1].base}
    for key, origin in origins.items():
        assert _affinity_origin(spark, key) == origin


def test_affinity_yields_when_its_preferred_replica_is_gone(pool) -> None:
    spark, thor = pool[0], pool[1]
    peer_key = next(
        (
            key
            for key in (f"sess-{i}" for i in range(40))
            if _affinity_origin(spark, key) == thor.base
        ),
        None,
    )
    assert peer_key is not None, "no affinity key preferred the peer replica"
    _shutdown(thor.httpd)
    pool.wait_ready(expect=False)
    with _pool_chat(spark, headers={S.AFFINITY_HEADER: peer_key}) as resp:
        assert resp.status == 200
        assert resp.headers.get(S.SERVED_BY_HEADER) == spark.base
        assert resp.headers.get(S.ROUTE_REASON_HEADER) != REASON_AFFINITY


# --- pairwise auth across a pooled forward ----------------------------------


def test_a_pooled_forward_carries_the_pairwise_key_and_never_the_callers() -> None:
    peer_key = "sk-pool-peer-inbound-5150"
    caller_key = "sk-pool-caller-inbound-6270"
    with _n_gateways(
        2,
        pool_env=[
            {"GATEWAY_API_KEY": caller_key, "PRIMARY_PEER_API_KEYS": peer_key},
            {"GATEWAY_API_KEY": peer_key},
        ],
    ) as p:
        p.wait_ready()
        spark, thor = p[0], p[1]
        p.backends[0].running = 9
        p.wait_ready()
        with _pool_chat(spark, key=caller_key) as resp:
            assert resp.status == 200
            assert resp.headers.get(S.PROXIED_BY_HEADER) == thor.base
        forwarded = _posts(thor)
        assert len(forwarded) == 1
        sent = {k.lower(): v for k, v in forwarded[0].headers}
        assert sent["authorization"] == f"Bearer {peer_key}"
        assert caller_key not in json.dumps(forwarded[0].headers)
        # And the inbound gate is real: an unkeyed caller never reaches the pool.
        _expect_error(
            401, spark.base, _CHAT_PATH, method="POST", body=_chat_body("cortex"), key=None
        )
        assert len(_posts(thor)) == 1


# --- (7) the no-pool golden -------------------------------------------------
#
# h1's byte-identity claim, pinned as a FILE rather than as literals inline:
# the fixture is generated by this very code path WITH the pool code present,
# so a regression that leaks a pool marker (or a fingerprint/replicas key)
# onto a no-pool deployment moves the diff, and the companion assertion below
# proves none of the five pool headers appear on any of the five responses.

NO_POOL_GOLDEN = Path(__file__).resolve().parent / "goldens" / "no-pool-gateway.json"

_NO_POOL_REGEN = "uv run python tests/goldens/regen.py"

# Headers whose value is a function of when/where the test ran, not of the
# gateway's behaviour. Content-Length rides along because the normalised body
# below no longer has the length the wire carried.
_VOLATILE_HEADERS = frozenset({"date", "server", "content-length"})

_POOL_HEADERS = (
    S.SERVED_BY_HEADER,
    S.PROXIED_BY_HEADER,
    S.ROUTE_REASON_HEADER,
    S.ROUTE_ATTEMPTS_HEADER,
    S.AFFINITY_HEADER,
)

#: ``(name, method, path, model)`` — the fixed request list the golden pins.
NO_POOL_REQUESTS: tuple[tuple[str, str, str, str | None], ...] = (
    ("chat-by-alias", "POST", _CHAT_PATH, "cortex"),
    ("chat-by-raw-id", "POST", _CHAT_PATH, _POOL_CORTEX_ID),
    ("chat-unknown-model", "POST", _CHAT_PATH, "no-such-model-anywhere"),
    ("v1-models", "GET", _MODELS_PATH, None),
    ("capabilities", "GET", "/capabilities", None),
)


def _no_pool_exchange(base: str, method: str, path: str, model: str | None) -> dict:
    """One request against the no-pool gateway, reduced to its comparable shape."""
    body = _chat_body(model) if model is not None else None
    try:
        resp = _request(
            base, path, method=method, body=body, key=None, headers={"Host": _GOLDEN_HOST}
        )
        raw = resp.read()
        resp.close()
        status, headers = resp.status, resp.headers
    except urllib.error.HTTPError as err:
        raw, status, headers = err.read(), err.code, err.headers
    return {
        "status": status,
        "headers": sorted(
            [key, value] for key, value in headers.items() if key.lower() not in _VOLATILE_HEADERS
        ),
        "body": json.loads(raw),
    }


#: ``capabilities_payload`` reads the served-context overlay off ``os.environ``
#: on the HTTP route, so an operator shell that happens to export one of these
#: would move the golden. Scrubbed inside the capture itself (not in a pytest
#: fixture) so ``regen.py`` — which has no fixtures — captures the same bytes.
_CONTEXT_OVERLAY_VARS = (
    "PRIMARY_MAX_MODEL_LEN",
    "MULTIMODAL_MAX_MODEL_LEN",
    "EMBED_MAX_MODEL_LEN",
    "RERANK_MAX_MODEL_LEN",
)


def capture_no_pool_golden() -> dict:
    """Drive :data:`NO_POOL_REQUESTS` against a gateway with NO pool declared.

    Called by the test below AND by ``tests/goldens/regen.py`` — one capture
    function, so the committed fixture can never disagree with what the test
    compares against.
    """
    backend = _FakeBackend(_POOL_CORTEX_ID, max_model_len=_POOL_MAX_MODEL_LEN)
    env = {
        "PRIMARY_URL": backend.base,
        "PRIMARY_SERVED_NAME": _POOL_CORTEX_ID,
    }
    # patch.dict snapshots the whole mapping and restores it on exit, so the
    # pops below are undone even if the capture raises.
    with mock.patch.dict(os.environ, {}, clear=False):
        for var in _CONTEXT_OVERLAY_VARS:
            os.environ.pop(var, None)
        # An IDLE pressure provider, not None: without one `handle_post`
        # skips the tier branch entirely, and the by-alias request would pin a
        # response that no deployed gateway (which always has a PressureCache)
        # actually produces.
        box = _spawn_gateway(env, pressure=_ManualPressure())
        try:
            box.cache.refresh()
            return {
                name: _no_pool_exchange(box.base, method, path, model)
                for name, method, path, model in NO_POOL_REQUESTS
            }
        finally:
            _shutdown(box.httpd, backend)


@pytest.fixture
def no_pool_capture():
    return capture_no_pool_golden()


def test_no_pool_gateway_matches_the_committed_golden(no_pool_capture) -> None:
    assert NO_POOL_GOLDEN.is_file(), f"missing golden {NO_POOL_GOLDEN} — run {_NO_POOL_REGEN}"
    expected = json.loads(NO_POOL_GOLDEN.read_text(encoding="utf-8"))
    assert no_pool_capture == expected, (
        f"{NO_POOL_GOLDEN.name} drifted — a no-pool deployment's wire bytes changed.\n"
        f"If that is deliberate, regenerate with `{_NO_POOL_REGEN}` and review the diff."
    )


def test_no_pool_gateway_emits_no_pool_headers_and_no_replica_keys(no_pool_capture) -> None:
    lowered = {header.lower() for header in _POOL_HEADERS}
    for name, exchange in no_pool_capture.items():
        present = {key.lower() for key, _value in exchange["headers"]} & lowered
        assert not present, f"{name} leaked pool markers: {sorted(present)}"
    payload = no_pool_capture["capabilities"]["body"]
    for role, entry in payload.items():
        assert "replicas" not in entry, f"{role} gained a replicas key with no pool declared"
        assert "fingerprint" not in entry, f"{role} gained a fingerprint key with no pool declared"
