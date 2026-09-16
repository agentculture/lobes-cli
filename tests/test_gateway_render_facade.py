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

import json
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

    def _record(self) -> None:
        type(self).paths.append(f"{self.command} {self.path}")

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


def test_the_caller_bearer_is_never_forwarded_to_comfyui(gateway) -> None:
    """ComfyUI has no auth; forwarding the fleet bearer into it would leak the
    key into a process that neither needs nor checks it."""
    forwarded: list[str] = []
    real = S.open_upstream

    def spy(backend, path, body, headers, **kw):
        forwarded.extend(k.lower() for k, _v in headers)
        return real(backend, path, body, headers, **kw)

    S.open_upstream = spy  # noqa: S3010 - restored below
    try:
        _submit(gateway)
    finally:
        S.open_upstream = real
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
