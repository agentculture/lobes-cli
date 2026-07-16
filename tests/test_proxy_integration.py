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
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from lobes.gateway import server as S
from lobes.gateway._config import build_config
from lobes.gateway._readiness import ReadinessCache

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

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._record()
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
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
        else:
            self._send_json(200, _chat_completion_body(self.server.served_id))


class _FakeBackend(ThreadingHTTPServer):
    """One fake vLLM engine on an ephemeral loopback port."""

    def __init__(self, served_id: str) -> None:
        self.served_id = served_id
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


def _spawn_gateway(env: dict[str, str], *, log: list | None = None) -> SimpleNamespace:
    """A REAL gateway: build_config → peer specs → a real ReadinessCache
    (``start=False``; tests seed it deterministically via ``refresh()``) → the
    real handler dispatch on a real ``ThreadingHTTPServer``. Nothing is
    monkeypatched — ``open_upstream`` opens genuine sockets."""
    table, cfg = build_config(env)
    specs = S.peer_specs_from_table(table, env)
    cache = ReadinessCache.from_backends(
        table.backends, peer_specs=tuple(specs.values()), start=False
    )
    handler = S._make_handler(table, cfg, None, cache, specs)
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
    assert len(posts) == 1 and posts[0].path == "/v1/chat/completions"
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
    assert "proxied" not in caps["cortex"] and "hosted_by" not in caps["cortex"]
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
    assert _CALLER_KEY not in captured.err and _CALLER_KEY not in captured.out
    assert _PEER_KEY not in captured.err and _PEER_KEY not in captured.out


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
        "responsibilities": [
            "reasoning",
            "deciding",
            "planning",
            "tool_use",
            "code_repo_actions",
            "validation",
            "final_authority",
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
        "context": 131072,
        "quant": "compressed-tensors",
        "mtp": True,
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
    "embedder": {
        "role": "embedder",
        "model": _EMBED_ID,
        "runtime": "vllm",
        "endpoint": _GOLDEN_ORIGIN,
        "path": "/v1/embeddings",
        "context": 32768,
        "quant": "",
        "mtp": False,
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
