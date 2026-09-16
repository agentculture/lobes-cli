"""The ``/v1/render`` facade (issue #82, t9) — job-scoped routes + fan-out.

Three acceptance criteria drive this file, and each one is proven by a
NEGATIVE control rather than by reading the code:

1. **The family is spelled under ``/v1/`` and wired into BOTH verb chains.**
   ``do_GET`` gates the bearer only for routes under ``/v1/`` (server.py's
   GET auth branch), so a render family spelled anywhere else would ship
   unauthenticated. That is asserted by driving a REAL gateway with
   ``GATEWAY_API_KEY`` set and watching an unauthenticated
   ``GET /v1/render/jobs/<id>`` come back 401 — not by inspecting the prefix.

2. **The facade is job-scoped.** The gateway mints its own opaque job id on
   submit and keeps the ComfyUI ``prompt_id`` to itself; status, artifact
   index, artifact bytes and cancel are served ONLY for ids it issued. An
   authorized caller naming an id it did not submit is refused 404, and a
   caller naming a real ComfyUI ``prompt_id`` (which is what a leaked
   ``/history`` listing would hand them) is refused too.

3. **The fan-out reaches submit / poll / artifact-fetch without exposing
   ``/history`` or ``/view`` enumeration outward.** The stub upstream records
   every path it is asked for, so the test can assert both what WAS dialed
   and that ``/history`` never was; and the outward route table is asserted to
   have no spelling that reaches ``/view`` with caller-chosen query
   parameters.

Scope honesty: the upstream here is a stdlib stub that mimics ComfyUI
0.33.2's shapes (``POST /prompt`` → ``prompt_id``, ``GET /api/jobs/<id>`` →
an outputs tree, ``GET /view`` → bytes). Nothing here proves the real
ComfyUI's schema; that is the live acceptance run (t13).
"""

from __future__ import annotations

import contextlib
import http.client
import io
import json
import socket
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from lobes.gateway import server as S
from lobes.gateway._config import build_config
from lobes.gateway._routing import (
    RENDER_PATH,
    RENDER_TASK,
    extract_job_artifacts,
    is_render_path,
    list_models_payload,
    parse_render_route,
    resolve_model,
)
from lobes.roles import _INNEREYE_MODEL

_API_KEY = "sk-render-test"
_PROMPT_ID = "comfy-prompt-0001"
_ARTIFACT = "flux_output_00001_.png"
_OTHER_ARTIFACT = "flux_output_00002_.png"  # another agent's output; never ours
_BLOB = bytes(range(256)) * 512  # 128 KiB, deliberately not valid UTF-8


# ===========================================================================
# A stub ComfyUI 0.33.2
# ===========================================================================


class _StubComfy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    paths: list[str] = []
    # Paths this stub answers by sending HEADERS and then dying mid-body — the
    # exact failure finding 4 names (ComfyUI returns headers, then resets).
    break_paths: set[str] = set()

    def _record(self) -> None:
        type(self).paths.append(f"{self.command} {self.path}")

    def _broken(self) -> None:
        """Headers, a truncated body, then a hard close.

        ``Content-Length`` promises far more than is written and the connection
        is torn down, so the gateway's buffered ``read_all()`` raises
        ``http.client.IncompleteRead`` — a response-phase failure, well after
        ``open_upstream`` returned.
        """
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "4096")
        self.end_headers()
        self.wfile.write(b'{"prompt')
        self.wfile.flush()
        self.close_connection = True

    def _is_broken(self) -> bool:
        return self.path.split("?", 1)[0] in type(self).break_paths

    def _json(self, status: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        self._record()
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self._is_broken():
            self._broken()
            return
        route = self.path.split("?", 1)[0]
        if route == "/prompt":
            self._json(200, {"prompt_id": _PROMPT_ID, "number": 3, "node_errors": {}})
        elif route == f"/api/jobs/{_PROMPT_ID}/cancel":
            self._json(200, {"cancelled": True})
        elif route == "/upload/image":
            self._json(200, {"name": "in.png", "subfolder": "", "type": "input"})
        else:
            self._json(404, {"error": "no such route"})

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        self._record()
        if self._is_broken():
            self._broken()
            return
        route, _, query = self.path.partition("?")
        if route == f"/api/jobs/{_PROMPT_ID}":
            self._json(
                200,
                {
                    "prompt_id": _PROMPT_ID,
                    "status": {"completed": True},
                    "outputs": {
                        "9": {
                            "images": [{"filename": _ARTIFACT, "subfolder": "", "type": "output"}]
                        }
                    },
                },
            )
            return
        if route == "/view":
            params = urllib.parse.parse_qs(query)
            if params.get("filename", [""])[0] != _ARTIFACT:
                self._json(404, {"error": "no such file"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(_BLOB)))
            self.end_headers()
            self.wfile.write(_BLOB)
            return
        self._json(404, {"error": "no such route"})

    def log_message(self, *a):
        pass


class _QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):  # pragma: no cover - noise only
        pass


@pytest.fixture
def comfy():
    _StubComfy.paths = []
    _StubComfy.break_paths = set()
    httpd = _QuietServer(("127.0.0.1", 0), _StubComfy)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    host, port = httpd.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _env(comfy_url: str | None, **over) -> dict:
    env = {
        "PRIMARY_SERVED_NAME": "P",
        "GATEWAY_DEFAULT_MODEL": "P",
        "GATEWAY_API_KEY": _API_KEY,
    }
    if comfy_url:
        env["INNEREYE_BASE_URL"] = comfy_url
    env.update(over)
    return env


@pytest.fixture
def gateway(comfy):
    table, cfg = build_config(_env(comfy))
    httpd = _QuietServer(("127.0.0.1", 0), S._make_handler(table, cfg))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    host, port = httpd.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


@contextlib.contextmanager
def _serve(comfy_url: str | None, **over):
    """A live gateway over an ad-hoc env — the `gateway` fixture, parameterised."""
    table, cfg = build_config(_env(comfy_url, **over))
    httpd = _QuietServer(("127.0.0.1", 0), S._make_handler(table, cfg))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    host, port = httpd.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _call(url, *, method="GET", body=None, key=_API_KEY, headers=None):
    """Returns (status, headers, bytes) — an HTTPError is a result, not a raise."""
    req = urllib.request.Request(url, data=body, method=method)
    if key is not None:
        req.add_header("Authorization", f"Bearer {key}")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:  # noqa: PERF203 - the point of the call
        return exc.code, dict(exc.headers), exc.read()


def _submit(gateway_url) -> str:
    status, _h, raw = _call(
        gateway_url + RENDER_PATH,
        method="POST",
        body=json.dumps({"prompt": {"1": {"class_type": "KSampler"}}}).encode(),
        headers={"Content-Type": "application/json"},
    )
    assert status == 200, raw
    return json.loads(raw)["job_id"]


# ===========================================================================
# AC1 — the family is under /v1/ and BOTH verb chains reach it; 401 negative
# ===========================================================================


def test_render_family_is_spelled_under_v1() -> None:
    assert RENDER_PATH == "/v1/render"
    assert RENDER_PATH.startswith("/v1/")
    assert is_render_path("/v1/render")
    assert is_render_path("/v1/render/jobs/abc")
    assert is_render_path("/v1/render/jobs/abc?x=1")
    assert not is_render_path("/v1/renderer")
    assert not is_render_path("/render")


def test_unauthenticated_get_under_v1_render_is_refused_401(gateway) -> None:
    """The load-bearing negative control for c2/h12 — measured, not assumed."""
    status, headers, raw = _call(gateway + "/v1/render/jobs/whatever", key=None)
    assert status == 401
    assert "WWW-Authenticate" in headers
    assert json.loads(raw)["error"]["code"] == "invalid_api_key"


def test_unauthenticated_post_to_v1_render_is_refused_401(gateway) -> None:
    status, _h, raw = _call(gateway + RENDER_PATH, method="POST", body=b"{}", key=None)
    assert status == 401
    assert json.loads(raw)["error"]["code"] == "invalid_api_key"


def test_unauthenticated_render_request_never_dials_comfyui(gateway) -> None:
    """A rejected caller costs zero upstream sockets — ComfyUI has NO auth of
    its own, so anything that reached it would be served."""
    _call(gateway + "/v1/render/jobs/whatever", key=None)
    _call(gateway + RENDER_PATH, method="POST", body=b"{}", key=None)
    assert _StubComfy.paths == []


def test_authenticated_get_reaches_the_facade_not_a_404(gateway) -> None:
    """Proves the 401 above came from the AUTH gate, not from an unwired route
    (an unrouted path would 404 for an authenticated caller too)."""
    status, _h, raw = _call(gateway + "/v1/render/jobs/whatever")
    assert status == 404
    assert json.loads(raw)["error"]["type"] == "render_job_not_found"


# ===========================================================================
# AC2 — job scoping
# ===========================================================================


def test_submit_issues_a_gateway_job_id_that_is_not_the_prompt_id(gateway) -> None:
    job_id = _submit(gateway)
    assert job_id
    assert job_id != _PROMPT_ID
    # The upstream id never travels outward — a caller cannot learn it and so
    # cannot name another agent's job by guessing ComfyUI's own id space.
    status, _h, raw = _call(gateway + f"/v1/render/jobs/{job_id}")
    assert status == 200
    assert _PROMPT_ID not in raw.decode()


def test_status_is_served_only_for_an_issued_id(gateway) -> None:
    job_id = _submit(gateway)
    assert _call(gateway + f"/v1/render/jobs/{job_id}")[0] == 200
    # Same shape of id, never issued here.
    status, _h, raw = _call(gateway + "/v1/render/jobs/deadbeefdeadbeef")
    assert status == 404
    assert json.loads(raw)["error"]["type"] == "render_job_not_found"


def test_a_real_comfy_prompt_id_is_refused(gateway) -> None:
    """The exact leak c35 names: a caller holding a /history listing knows real
    prompt ids. The facade refuses them — only ids IT issued are addressable."""
    _submit(gateway)
    before = list(_StubComfy.paths)
    status, _h, raw = _call(gateway + f"/v1/render/jobs/{_PROMPT_ID}")
    assert status == 404
    assert json.loads(raw)["error"]["type"] == "render_job_not_found"
    # ...and the refusal never dialed ComfyUI at all.
    assert _StubComfy.paths == before


def test_artifact_bytes_are_served_only_for_an_issued_id(gateway) -> None:
    job_id = _submit(gateway)
    ok = _call(gateway + f"/v1/render/jobs/{job_id}/artifacts/{_ARTIFACT}")
    assert ok[0] == 200
    assert ok[2] == _BLOB
    bad = _call(gateway + f"/v1/render/jobs/deadbeef/artifacts/{_ARTIFACT}")
    assert bad[0] == 404
    assert json.loads(bad[2])["error"]["type"] == "render_job_not_found"


def test_a_filename_this_job_did_not_produce_is_refused(gateway) -> None:
    """The counter-name enumeration c35 names: an authorized caller with a
    valid job id still cannot reach the NEXT counter's file."""
    job_id = _submit(gateway)
    status, _h, raw = _call(gateway + f"/v1/render/jobs/{job_id}/artifacts/{_OTHER_ARTIFACT}")
    assert status == 404
    assert json.loads(raw)["error"]["type"] == "render_artifact_not_found"
    # /view was never dialed for the foreign name.
    assert not any(_OTHER_ARTIFACT in p for p in _StubComfy.paths)


def test_cancel_is_served_only_for_an_issued_id(gateway) -> None:
    job_id = _submit(gateway)
    assert _call(gateway + f"/v1/render/jobs/{job_id}/cancel", method="POST", body=b"")[0] == 200
    bad = _call(gateway + "/v1/render/jobs/deadbeef/cancel", method="POST", body=b"")
    assert bad[0] == 404
    assert json.loads(bad[2])["error"]["type"] == "render_job_not_found"


def test_artifact_index_is_job_scoped(gateway) -> None:
    job_id = _submit(gateway)
    status, _h, raw = _call(gateway + f"/v1/render/jobs/{job_id}/artifacts")
    assert status == 200
    payload = json.loads(raw)
    assert payload["job_id"] == job_id
    assert [a["filename"] for a in payload["artifacts"]] == [_ARTIFACT]


def test_the_registry_forgets_nothing_it_issued_under_its_limit() -> None:
    reg = S.RenderJobRegistry(limit=3)
    ids = [reg.issue(f"p{i}") for i in range(3)]
    assert [reg.prompt_id(i) for i in ids] == ["p0", "p1", "p2"]
    # ...and evicts oldest-first past the limit, failing CLOSED (a forgotten id
    # is refused, never served from a stale or guessed mapping).
    extra = reg.issue("p3")
    assert reg.prompt_id(ids[0]) is None
    assert reg.prompt_id(extra) == "p3"


def test_the_registry_is_thread_safe() -> None:
    reg = S.RenderJobRegistry(limit=10_000)
    issued: list[str] = []
    lock = threading.Lock()

    def work(n: int) -> None:
        mine = [reg.issue(f"p{n}-{i}") for i in range(50)]
        with lock:
            issued.extend(mine)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(issued)) == 400  # every id unique, none lost
    assert all(reg.prompt_id(i) is not None for i in issued)


# ===========================================================================
# AC3 — the fan-out, and no /history or /view enumeration outward
# ===========================================================================


def test_fan_out_reaches_submit_poll_and_artifact_fetch(comfy, gateway) -> None:
    job_id = _submit(gateway)
    _call(gateway + f"/v1/render/jobs/{job_id}")
    _call(gateway + f"/v1/render/jobs/{job_id}/artifacts/{_ARTIFACT}")
    dialed = list(_StubComfy.paths)
    assert "POST /prompt" in dialed
    assert f"GET /api/jobs/{_PROMPT_ID}" in dialed
    assert any(p.startswith("GET /view?") for p in dialed)


def test_the_fan_out_never_dials_history(gateway) -> None:
    job_id = _submit(gateway)
    _call(gateway + f"/v1/render/jobs/{job_id}")
    _call(gateway + f"/v1/render/jobs/{job_id}/artifacts")
    _call(gateway + f"/v1/render/jobs/{job_id}/artifacts/{_ARTIFACT}")
    assert not any("/history" in p or "/queue" in p for p in _StubComfy.paths)


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/v1/render/history"),
        ("GET", "/v1/render/history/abc"),
        ("GET", "/v1/render/view?filename=flux_output_00002_.png"),
        ("GET", "/v1/render/queue"),
        ("GET", "/v1/render/jobs"),
        ("GET", "/v1/render"),
        ("GET", "/v1/render/jobs/../../history"),
        ("POST", "/v1/render/jobs/abc/interrupt"),
    ],
)
def test_no_outward_route_spells_an_enumerable_comfy_surface(method, path) -> None:
    assert parse_render_route(path, method) is None


def test_an_enumerable_spelling_404s_through_the_live_gateway(gateway) -> None:
    status, _h, raw = _call(gateway + "/v1/render/view?filename=" + _OTHER_ARTIFACT)
    assert status == 404
    assert json.loads(raw)["error"]["type"] == "not_found"
    assert _StubComfy.paths == []


def test_artifact_bytes_are_relayed_byte_for_byte(comfy, gateway) -> None:
    job_id = _submit(gateway)
    status, headers, through = _call(gateway + f"/v1/render/jobs/{job_id}/artifacts/{_ARTIFACT}")
    assert status == 200
    assert through == _BLOB
    assert headers.get("Content-Type") == "image/png"


def test_upload_is_relayed_for_input_images(gateway) -> None:
    status, _h, raw = _call(
        gateway + "/v1/render/uploads/image",
        method="POST",
        body=b"--b\r\nContent-Disposition: form-data; name=image\r\n\r\nx\r\n--b--\r\n",
        headers={"Content-Type": "multipart/form-data; boundary=b"},
    )
    assert status == 200
    assert json.loads(raw)["type"] == "input"
    assert "POST /upload/image" in _StubComfy.paths


def test_the_caller_bearer_is_never_forwarded_to_comfyui(gateway, monkeypatch) -> None:
    """ComfyUI has no auth; forwarding the fleet bearer into it would leak the
    key into a process that neither needs nor checks it."""
    forwarded: list[str] = []
    real = S.open_upstream

    def spy(backend, path, body, headers, **kw):
        forwarded.extend(k.lower() for k, _v in headers)
        return real(backend, path, body, headers, **kw)

    # `monkeypatch` owns the restore: the module attribute is global state, and
    # a hand-rolled try/finally only unwinds for the paths it wraps.
    monkeypatch.setattr(S, "open_upstream", spy)
    _submit(gateway)
    assert "authorization" not in forwarded


# ===========================================================================
# Wiring: the INNEREYE_BASE_URL reader, and the lane's model-routing posture
# ===========================================================================


def test_innereye_backend_is_wired_only_by_its_base_url() -> None:
    table, _cfg = build_config(_env(None))
    assert not [b for b in table.backends if b.name == "innereye"]
    assert "innereye" in table.infeasible  # OPT_IN_BACKENDS: unwired ⇒ infeasible

    table, _cfg = build_config(_env("http://comfyui:8188"))
    wired = [b for b in table.backends if b.name == "innereye"]
    assert len(wired) == 1
    assert wired[0].base_url == "http://comfyui:8188"
    assert wired[0].served_name == _INNEREYE_MODEL
    assert wired[0].task == RENDER_TASK
    assert "innereye" not in table.infeasible


def test_the_render_lane_is_path_routed_never_model_routed() -> None:
    """It is a ComfyUI tenant, not an OpenAI model: its id must not be
    addressable through a `model` field nor advertised on /v1/models."""
    table, _cfg = build_config(_env("http://comfyui:8188"))
    ids = {e["id"] for e in list_models_payload(table)["data"]}
    assert _INNEREYE_MODEL not in ids
    ready = {b.name: True for b in table.backends}
    assert _INNEREYE_MODEL not in {e["id"] for e in list_models_payload(table, ready)["data"]}
    # ...and a chat request naming it falls back to the default model rather
    # than resolving to the render backend.
    assert resolve_model(table, _INNEREYE_MODEL) == "P"


def test_render_requests_404_role_infeasible_when_the_lane_is_unwired() -> None:
    table, cfg = build_config(_env(None))
    httpd = _QuietServer(("127.0.0.1", 0), S._make_handler(table, cfg))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    host, port = httpd.server_address
    try:
        base = f"http://{host}:{port}"
        for status, _h, raw in (
            _call(base + RENDER_PATH, method="POST", body=b"{}"),
            _call(base + "/v1/render/jobs/abc"),
        ):
            assert status == 404
            assert json.loads(raw)["error"]["type"] == "role_infeasible"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_an_unreachable_comfyui_yields_a_retryable_503(comfy) -> None:
    table, cfg = build_config(_env("http://127.0.0.1:1", GATEWAY_CONNECT_TIMEOUT="1"))
    httpd = _QuietServer(("127.0.0.1", 0), S._make_handler(table, cfg))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    host, port = httpd.server_address
    try:
        status, headers, raw = _call(
            f"http://{host}:{port}" + RENDER_PATH, method="POST", body=b"{}"
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert status == 503
    assert headers.get("Retry-After") == str(S.RENDER_RETRY_AFTER_SECONDS)
    payload = json.loads(raw)["error"]
    assert payload["type"] == "render_backend_unavailable"
    message = payload["message"]
    # AC2 (t10): the body must be an HONEST warming response, never a claim
    # that a boot is (or will be) triggered automatically -- lobes has no
    # lifecycle actuator in the data plane (see
    # test_gateway_no_lifecycle_actuator.py). It should instead point at the
    # real, manual remedy.
    assert "warming" in message
    assert "lobes never starts it automatically" in message
    assert "lobes up innereye --apply" in message


# ===========================================================================
# The pure helpers
# ===========================================================================


@pytest.mark.parametrize(
    "method,path,kind",
    [
        ("POST", "/v1/render", "submit"),
        ("POST", "/v1/render/jobs/abc/cancel", "cancel"),
        ("POST", "/v1/render/uploads/image", "upload"),
        ("GET", "/v1/render/jobs/abc", "status"),
        ("GET", "/v1/render/jobs/abc/artifacts", "artifacts"),
        ("GET", "/v1/render/jobs/abc/artifacts/out.png", "artifact"),
    ],
)
def test_parse_render_route_kinds(method, path, kind) -> None:
    route = parse_render_route(path, method)
    assert route is not None
    assert route.kind == kind


def test_parse_render_route_rejects_a_malformed_job_id() -> None:
    assert parse_render_route("/v1/render/jobs/a%2Fb", "GET") is None
    assert parse_render_route("/v1/render/jobs/" + "x" * 200, "GET") is None
    assert parse_render_route("/v1/render/jobs/a b", "GET") is None


def test_extract_job_artifacts_is_tolerant_of_both_comfy_shapes() -> None:
    api_jobs = {"outputs": {"9": {"images": [{"filename": "a.png", "subfolder": "s"}]}}}
    history = {_PROMPT_ID: {"outputs": {"9": {"images": [{"filename": "a.png", "type": "temp"}]}}}}
    assert extract_job_artifacts(api_jobs) == (
        {"filename": "a.png", "subfolder": "s", "type": "output"},
    )
    assert extract_job_artifacts(history) == (
        {"filename": "a.png", "subfolder": "", "type": "temp"},
    )
    assert extract_job_artifacts(None) == ()
    assert extract_job_artifacts({"outputs": {}}) == ()


def test_extract_job_artifacts_dedupes_and_is_bounded() -> None:
    dupes = {"o": [{"filename": "a.png"}, {"filename": "a.png"}, {"filename": "b.png"}]}
    assert [a["filename"] for a in extract_job_artifacts(dupes)] == ["a.png", "b.png"]
    many = {"o": [{"filename": f"{i}.png"} for i in range(2000)]}
    assert len(extract_job_artifacts(many)) <= 512


# ===========================================================================
# Request-body size limits on the render lane (review finding 1)
# ===========================================================================
#
# Qodo, HIGH: ``do_POST`` reads the complete request body without a size limit
# and ``_render_upload`` hands that in-memory value straight to ComfyUI's
# WRITABLE upload endpoint, so any admitted caller can exhaust gateway memory
# (and then backend storage) with one arbitrarily large fixed-length or chunked
# image.
#
# The fix is deliberately RENDER-SCOPED. The unbounded read is pre-existing and
# common to every POST route (chat completions, /v1/audio/* multipart); a global
# cap is a separate change with its own blast radius. So the mechanism is
# general (``_read_body(limit)`` + a strict ``read_chunked_body``) but its
# DEFAULT is inert — ``_post_body_limit`` answers ``None`` everywhere except the
# render family, which is asserted below rather than asserted about the code.


def _post_chunked(base, path, chunks, *, key=_API_KEY, ctype="application/json"):
    """POST a ``Transfer-Encoding: chunked`` body from a raw socket.

    urllib cannot spell chunked, and the point of this helper is to control the
    framing exactly: a refusal must land even though no ``Content-Length`` ever
    declared the size.
    """
    parts = urllib.parse.urlsplit(base)
    sock = socket.create_connection((parts.hostname, parts.port), timeout=30)
    try:
        head = (
            f"POST {path} HTTP/1.1\r\n"
            f"Host: {parts.hostname}:{parts.port}\r\n"
            f"Authorization: Bearer {key}\r\n"
            f"Content-Type: {ctype}\r\n"
            "Transfer-Encoding: chunked\r\n\r\n"
        )
        sock.sendall(head.encode())
        try:
            for chunk in chunks:
                sock.sendall(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
            sock.sendall(b"0\r\n\r\n")
        except OSError:
            # The gateway may refuse and close before the last chunk lands —
            # that IS the incremental enforcement, not a test failure.
            pass
        resp = http.client.HTTPResponse(sock)
        resp.begin()
        return resp.status, dict(resp.getheaders()), resp.read()
    finally:
        sock.close()


def test_the_body_limit_defaults_are_render_sized() -> None:
    _table, cfg = build_config(_env("http://comfyui:8188"))
    # A ComfyUI API-format graph is tens of KB; 1 MiB is ~30x headroom.
    assert cfg.render_max_workflow_bytes == 1024 * 1024
    # An input image is legitimately several MB; 32 MiB is generous and still
    # strictly below read_chunked_body's pre-existing 64 MiB ceiling.
    assert cfg.render_max_upload_bytes == 32 * 1024 * 1024


def test_the_body_limits_are_operator_configurable() -> None:
    _table, cfg = build_config(
        _env(
            "http://comfyui:8188",
            GATEWAY_RENDER_MAX_WORKFLOW_BYTES="4096",
            GATEWAY_RENDER_MAX_UPLOAD_BYTES="8192",
        )
    )
    assert cfg.render_max_workflow_bytes == 4096
    assert cfg.render_max_upload_bytes == 8192


def test_an_oversized_fixed_length_workflow_is_refused_413(comfy) -> None:
    with _serve(comfy, GATEWAY_RENDER_MAX_WORKFLOW_BYTES="512") as base:
        status, headers, raw = _call(
            base + RENDER_PATH,
            method="POST",
            body=b"x" * 4096,
            headers={"Content-Type": "application/json"},
        )
    assert status == 413
    payload = json.loads(raw)["error"]
    assert payload["type"] == "render_payload_too_large"
    assert "GATEWAY_RENDER_MAX_WORKFLOW_BYTES" in payload["message"]
    # Rejected BEFORE the body was read, so the connection cannot be reused.
    assert headers.get("Connection", "").lower() == "close"
    # ...and nothing reached ComfyUI, in memory or on disk.
    assert _StubComfy.paths == []


def test_an_oversized_chunked_workflow_is_refused_413(comfy) -> None:
    """No ``Content-Length`` to pre-check: the cap must bite WHILE decoding."""
    with _serve(comfy, GATEWAY_RENDER_MAX_WORKFLOW_BYTES="512") as base:
        status, headers, raw = _post_chunked(base, RENDER_PATH, [b"y" * 256] * 8)
    assert status == 413
    assert json.loads(raw)["error"]["type"] == "render_payload_too_large"
    assert headers.get("Connection", "").lower() == "close"
    assert _StubComfy.paths == []


def test_an_oversized_fixed_length_upload_is_refused_413(comfy) -> None:
    with _serve(comfy, GATEWAY_RENDER_MAX_UPLOAD_BYTES="1024") as base:
        status, _h, raw = _call(
            base + "/v1/render/uploads/image",
            method="POST",
            body=b"z" * 8192,
            headers={"Content-Type": "multipart/form-data; boundary=b"},
        )
    assert status == 413
    payload = json.loads(raw)["error"]
    assert payload["type"] == "render_payload_too_large"
    assert "GATEWAY_RENDER_MAX_UPLOAD_BYTES" in payload["message"]
    assert "POST /upload/image" not in _StubComfy.paths


def test_an_oversized_chunked_upload_is_refused_413(comfy) -> None:
    with _serve(comfy, GATEWAY_RENDER_MAX_UPLOAD_BYTES="1024") as base:
        status, _h, raw = _post_chunked(
            base,
            "/v1/render/uploads/image",
            [b"z" * 512] * 8,
            ctype="multipart/form-data; boundary=b",
        )
    assert status == 413
    assert json.loads(raw)["error"]["type"] == "render_payload_too_large"
    assert "POST /upload/image" not in _StubComfy.paths


def test_the_upload_limit_is_larger_than_the_workflow_limit(comfy) -> None:
    """A body over the workflow cap but under the upload cap is an UPLOAD, and
    must still be served — the two limits are genuinely per-route."""
    with _serve(
        comfy,
        GATEWAY_RENDER_MAX_WORKFLOW_BYTES="512",
        GATEWAY_RENDER_MAX_UPLOAD_BYTES="65536",
    ) as base:
        big = b"--b\r\nContent-Disposition: form-data; name=image\r\n\r\n" + b"p" * 4096
        status, _h, raw = _call(
            base + "/v1/render/uploads/image",
            method="POST",
            body=big + b"\r\n--b--\r\n",
            headers={"Content-Type": "multipart/form-data; boundary=b"},
        )
        assert status == 200, raw
        assert json.loads(raw)["type"] == "input"
        # ...while the same size submitted as a WORKFLOW is refused.
        assert (
            _call(
                base + RENDER_PATH,
                method="POST",
                body=b"x" * 4096,
                headers={"Content-Type": "application/json"},
            )[0]
            == 413
        )


def test_a_body_within_the_limit_is_unaffected(comfy) -> None:
    with _serve(comfy, GATEWAY_RENDER_MAX_WORKFLOW_BYTES="65536") as base:
        assert _submit(base)
        # ...including a chunked one, which takes the incremental path.
        status, _h, raw = _post_chunked(
            base, RENDER_PATH, [json.dumps({"prompt": {"1": {}}}).encode()]
        )
    assert status == 200, raw
    assert json.loads(raw)["job_id"]


def test_a_non_positive_limit_disables_the_cap(comfy) -> None:
    """The documented escape hatch: 0 means "no limit", the pre-fix behaviour."""
    with _serve(comfy, GATEWAY_RENDER_MAX_WORKFLOW_BYTES="0") as base:
        status, _h, raw = _call(
            base + RENDER_PATH,
            method="POST",
            body=json.dumps({"prompt": {"1": {}}, "pad": "x" * 200_000}).encode(),
            headers={"Content-Type": "application/json"},
        )
    assert status == 200, raw


@pytest.mark.parametrize(
    "path",
    [
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/embeddings",
        "/v1/rerank",
        "/v1/score",
        "/v1/audio/transcriptions",
        "/v1/audio/speech",
        "/mesh/announce",
    ],
)
def test_no_non_render_post_route_carries_a_body_limit(path) -> None:
    """The scoping guarantee, asserted by construction: every lane but render
    answers None, so `_read_body` behaves exactly as it did before the fix."""
    table, cfg = build_config(_env("http://comfyui:8188"))
    handler = S._make_handler(table, cfg)
    assert S._Handler._post_body_limit(_FakeHandler(handler, path)) is None


@pytest.mark.parametrize(
    "path,attr",
    [
        ("/v1/render", "render_max_workflow_bytes"),
        ("/v1/render/uploads/image", "render_max_upload_bytes"),
        ("/v1/render/jobs/abc/cancel", "render_max_workflow_bytes"),
        # An unrouted render spelling is capped too: it is refused after the
        # body is read, so an uncapped read would still have happened.
        ("/v1/render/nope", "render_max_workflow_bytes"),
    ],
)
def test_every_render_post_route_carries_a_body_limit(path, attr) -> None:
    table, cfg = build_config(_env("http://comfyui:8188"))
    handler = S._make_handler(table, cfg)
    limit = S._Handler._post_body_limit(_FakeHandler(handler, path))
    assert limit == getattr(cfg, attr)


class _FakeHandler:
    """Just enough of a handler for `_post_body_limit`, which reads only
    `self.path` and `self.server_config`."""

    def __init__(self, handler_cls, path: str) -> None:
        self.path = path
        self.server_config = handler_cls.server_config


def test_the_default_chunked_reader_is_unchanged() -> None:
    """The general mechanism's inert default: without `strict`,
    `read_chunked_body` still TRUNCATES at its cap rather than raising, so
    every pre-existing caller behaves byte-identically."""
    wire = io.BytesIO(b"8\r\n" + b"a" * 8 + b"\r\n8\r\n" + b"b" * 8 + b"\r\n0\r\n\r\n")
    assert S.read_chunked_body(wire, 4) == b"a" * 8
    wire.seek(0)
    with pytest.raises(S.RequestBodyTooLarge):
        S.read_chunked_body(wire, 4, strict=True)


# ===========================================================================
# A mid-response upstream failure is the structured retryable 503 (finding 4)
# ===========================================================================
#
# Qodo, MEDIUM: `_render_response` translates only `UpstreamError`, but
# `_render_fetch` calls `read_all()` AFTER `open_upstream`'s wrapping block has
# returned. ComfyUI answering headers and then resetting therefore let the read
# exception escape the handler and abort the client connection, losing the
# documented structured answer.
#
# The same `503 render_backend_unavailable` + `Retry-After` is reused, not a new
# code: from the caller's side the two are the same fact (the render backend did
# not complete this request, try again shortly), and the existing body is the
# only place the "lobes never starts it automatically" honesty lives. The
# message names the mid-response case explicitly so the 503 is not claiming a
# cold backend it did not observe.


def _assert_retryable_503(status, headers, raw) -> None:
    assert status == 503
    assert headers.get("Retry-After") == str(S.RENDER_RETRY_AFTER_SECONDS)
    payload = json.loads(raw)["error"]
    assert payload["type"] == "render_backend_unavailable"
    assert "mid-response" in payload["message"]
    assert "lobes up innereye --apply" in payload["message"]


def test_a_midresponse_failure_on_submit_is_the_retryable_503(comfy, gateway) -> None:
    _StubComfy.break_paths = {"/prompt"}
    status, headers, raw = _call(
        gateway + RENDER_PATH,
        method="POST",
        body=json.dumps({"prompt": {"1": {}}}).encode(),
        headers={"Content-Type": "application/json"},
    )
    _assert_retryable_503(status, headers, raw)


def test_a_midresponse_failure_on_polling_is_the_retryable_503(comfy, gateway) -> None:
    job_id = _submit(gateway)
    _StubComfy.break_paths = {f"/api/jobs/{_PROMPT_ID}"}
    _assert_retryable_503(*_call(gateway + f"/v1/render/jobs/{job_id}"))


def test_a_midresponse_failure_on_the_artifact_index_is_the_retryable_503(comfy, gateway) -> None:
    job_id = _submit(gateway)
    _StubComfy.break_paths = {f"/api/jobs/{_PROMPT_ID}"}
    _assert_retryable_503(*_call(gateway + f"/v1/render/jobs/{job_id}/artifacts"))


def test_a_midresponse_failure_on_artifact_bytes_is_the_retryable_503(comfy, gateway) -> None:
    job_id = _submit(gateway)
    _StubComfy.break_paths = {f"/api/jobs/{_PROMPT_ID}"}
    _assert_retryable_503(*_call(gateway + f"/v1/render/jobs/{job_id}/artifacts/{_ARTIFACT}"))


def test_a_midresponse_failure_on_cancel_is_the_retryable_503(comfy, gateway) -> None:
    job_id = _submit(gateway)
    _StubComfy.break_paths = {f"/api/jobs/{_PROMPT_ID}/cancel"}
    _assert_retryable_503(
        *_call(gateway + f"/v1/render/jobs/{job_id}/cancel", method="POST", body=b"")
    )


def test_a_midresponse_failure_on_upload_is_the_retryable_503(comfy, gateway) -> None:
    _StubComfy.break_paths = {"/upload/image"}
    _assert_retryable_503(
        *_call(
            gateway + "/v1/render/uploads/image",
            method="POST",
            body=b"--b\r\n\r\nx\r\n--b--\r\n",
            headers={"Content-Type": "multipart/form-data; boundary=b"},
        )
    )


def test_the_client_connection_survives_a_midresponse_failure(comfy, gateway) -> None:
    """The regression in one line: before the fix the exception unwound out of
    the handler and the caller got a dropped socket instead of an answer. A
    SECOND request on the same gateway proves the handler unwound cleanly."""
    _StubComfy.break_paths = {"/prompt"}
    assert _call(gateway + RENDER_PATH, method="POST", body=b"{}")[0] == 503
    _StubComfy.break_paths = set()
    assert _submit(gateway)
