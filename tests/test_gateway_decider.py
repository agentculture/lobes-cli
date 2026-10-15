"""``POST /v1/systemone`` — the decider lane, routed by header, never shed.

The decider is a schema-scored decision model (Cloudflare Clef, the Strands
decider): its wire is the upstream ``/v1/systemone`` JSON ``{state,
questions}``, which carries NO ``model`` field. So the lane is PATH-routed,
like ``/v1/audio/*`` and ``/v1/render``, and the domain (which decider
checkpoint/adapter answers) is chosen by the ``X-Lobes-Model: decider:<name>``
header instead of the body. Every contract below is proven against a REAL
gateway relaying to a stdlib stub decider that records the raw bytes it got:

* the body reaches the backend BYTE-IDENTICAL (no ``rewrite_model``, no
  re-serialisation) and the backend's response comes back unchanged;
* no header resolves to ``decider:general``;
* an undeclared domain is 404 ``model_not_found`` (never a silent fall-back to
  ``general``); a box with no decider wired is 404 ``role_infeasible``;
* swap > 75 % and iowait > 50 % never shed a decider request (never 429);
* the route sits behind the ``GATEWAY_API_KEY`` bearer gate.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest

from lobes.gateway import server as S
from lobes.gateway._config import build_config
from lobes.gateway._pressure_policy import NEVER_SHED_ROLES
from lobes.gateway._routing import (
    DECIDE_TASK,
    DECIDER_MODEL_HEADER,
    SYSTEMONE_PATH,
    is_systemone_path,
    list_models_payload,
    model_routed_backends,
    resolve_decider_route,
)
from lobes.gateway._tier_request import PressureCache

_API_KEY = "sk-decider-test"

# Deliberately NOT what json.dumps would emit: odd spacing, key order a
# re-serialiser would keep but spacing it would not, a non-ASCII string a
# default json.dumps would \u-escape, and a trailing newline. Any parse+dump on
# the gateway path changes these bytes.
_BODY = (
    '{"state":  {"ticket": "réseau — down", "n": 3},\n'
    ' "questions": [{"id": "q1", "schema": {"type": "boolean"}}]}\n'
).encode("utf-8")

# The stub's answer is likewise oddly formatted so a re-serialising relay shows.
_ANSWER = b'{"answers" :[{"id":"q1","value":true,"score": 0.93}]}  '


class _StubDecider(BaseHTTPRequestHandler):
    """Records every request (path, raw body, headers) and answers ``_ANSWER``."""

    protocol_version = "HTTP/1.1"

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.server.seen.append(
            {
                "path": self.path,
                "body": raw,
                "headers": {k.lower(): v for k, v in self.headers.items()},
            }
        )
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(_ANSWER)))
        self.end_headers()
        self.wfile.write(_ANSWER)

    def log_message(self, *a):
        pass


class _QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):  # pragma: no cover - noise only
        pass


@contextlib.contextmanager
def _stub():
    httpd = _QuietServer(("127.0.0.1", 0), _StubDecider)
    httpd.seen = []
    threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    ).start()
    host, port = httpd.server_address
    try:
        yield httpd, f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


@contextlib.contextmanager
def _gateway(env: dict, *, pressure_cache: PressureCache | None = None):
    base = {"PRIMARY_SERVED_NAME": "P", "GATEWAY_DEFAULT_MODEL": "P"}
    base.update(env)
    table, cfg = build_config(base)
    httpd = _QuietServer(
        ("127.0.0.1", 0), S._make_handler(table, cfg, pressure_cache=pressure_cache)
    )
    threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    ).start()
    host, port = httpd.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _post(url: str, body: bytes = _BODY, headers: dict | None = None):
    parts = urlsplit(url)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=10)
    try:
        conn.request(
            "POST",
            SYSTEMONE_PATH,
            body=body,
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        resp = conn.getresponse()
        return resp.status, dict(resp.getheaders()), resp.read()
    finally:
        conn.close()


def _domains(**urls: str) -> str:
    return ",".join(f"{name}={url}" for name, url in urls.items())


# ===========================================================================
# Byte-for-byte forwarding and header routing (acceptance criterion 1)
# ===========================================================================


def test_body_reaches_the_backend_byte_identical_and_the_answer_comes_back_unchanged():
    with _stub() as (general, g_url), _stub() as (ops, o_url):
        env = {"DECIDER_DOMAINS": _domains(general=g_url, ops=o_url)}
        with _gateway(env) as gw:
            status, _headers, raw = _post(gw, headers={DECIDER_MODEL_HEADER: "decider:ops"})
    assert status == 200
    assert raw == _ANSWER
    assert general.seen == []
    assert len(ops.seen) == 1
    assert ops.seen[0]["body"] == _BODY
    assert ops.seen[0]["path"] == SYSTEMONE_PATH


def test_no_header_resolves_to_decider_general():
    with _stub() as (general, g_url), _stub() as (ops, o_url):
        env = {"DECIDER_DOMAINS": _domains(ops=o_url, general=g_url)}
        with _gateway(env) as gw:
            status, _headers, raw = _post(gw)
    assert status == 200
    assert raw == _ANSWER
    assert ops.seen == []
    assert [s["body"] for s in general.seen] == [_BODY]


def test_single_backend_wiring_is_domain_general():
    """Back-compat with d1: DECIDER_BASE_URL alone is the ``general`` domain."""
    with _stub() as (general, g_url):
        with _gateway({"DECIDER_BASE_URL": g_url}) as gw:
            status, _h, _raw = _post(gw, headers={DECIDER_MODEL_HEADER: "decider:general"})
            bare, _h2, _raw2 = _post(gw)
    assert (status, bare) == (200, 200)
    assert [s["body"] for s in general.seen] == [_BODY, _BODY]


# ===========================================================================
# Refusals (acceptance criterion 2)
# ===========================================================================


def test_undeclared_domain_is_model_not_found_and_never_falls_back_to_general():
    with _stub() as (general, g_url):
        with _gateway({"DECIDER_DOMAINS": _domains(general=g_url)}) as gw:
            status, _headers, raw = _post(gw, headers={DECIDER_MODEL_HEADER: "decider:legal"})
    assert status == 404
    error = json.loads(raw)["error"]
    assert error["code"] == "model_not_found"
    assert error["type"] == "model_not_found"
    assert "decider:legal" in error["message"]
    assert general.seen == []


@pytest.mark.parametrize("value", ["cortex", "decider:", "decider:Ops", "decider:../x", "hand:ops"])
def test_a_header_that_is_not_a_decider_spelling_is_model_not_found(value):
    with _stub() as (general, g_url):
        with _gateway({"DECIDER_DOMAINS": _domains(general=g_url)}) as gw:
            status, _headers, raw = _post(gw, headers={DECIDER_MODEL_HEADER: value})
    assert status == 404
    assert json.loads(raw)["error"]["code"] == "model_not_found"
    assert general.seen == []


def test_no_decider_wired_is_role_infeasible():
    with _gateway({}) as gw:
        status, _headers, raw = _post(gw)
        named, _h, named_raw = _post(gw, headers={DECIDER_MODEL_HEADER: "decider:ops"})
    assert (status, named) == (404, 404)
    for body in (raw, named_raw):
        error = json.loads(body)["error"]
        assert error["code"] == "role_infeasible"
        assert error["type"] == "role_infeasible"


def test_decider_declared_infeasible_is_role_infeasible_even_when_wired():
    with _stub() as (general, g_url):
        env = {"DECIDER_DOMAINS": _domains(general=g_url), "DECIDER_FEASIBLE": "false"}
        with _gateway(env) as gw:
            status, _headers, raw = _post(gw)
    assert status == 404
    assert json.loads(raw)["error"]["code"] == "role_infeasible"
    assert general.seen == []


def test_a_wired_but_unreachable_domain_is_a_retryable_503():
    """A down sidecar is "not now" (503 backend_unavailable), not "never"."""
    with _stub() as (_general, g_url):
        pass  # the stub is shut down: its port now refuses connections
    with _gateway({"DECIDER_DOMAINS": _domains(general=g_url)}) as gw:
        status, headers, raw = _post(gw)
    assert status == 503
    assert json.loads(raw)["error"]["type"] == "backend_unavailable"
    assert headers.get("Retry-After")


# ===========================================================================
# Never shed under pressure (acceptance criterion 2)
# ===========================================================================


def test_decider_is_never_shed_under_swap_and_iowait_pressure():
    assert "decider" in NEVER_SHED_ROLES
    busy = PressureCache(
        sampler=lambda: {"swap_used_percent": 95.0, "iowait_percent": 80.0}, start=False
    )
    with _stub() as (general, g_url):
        with _gateway({"DECIDER_DOMAINS": _domains(general=g_url)}, pressure_cache=busy) as gw:
            results = [_post(gw)[0] for _ in range(5)]
            # Negative control: the same box under the same pressure DOES shed
            # a full-tier chat request, so the pressure is real.
            parts = urlsplit(gw)
            conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=10)
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps({"model": "cortex", "messages": []}).encode(),
                headers={"Content-Type": "application/json"},
            )
            chat_status = conn.getresponse().status
            conn.close()
    assert results == [200] * 5
    assert len(general.seen) == 5
    assert chat_status == 429


# ===========================================================================
# The bearer gate (acceptance criterion 3)
# ===========================================================================


def test_the_route_sits_behind_the_gateway_api_key_bearer_gate():
    with _stub() as (general, g_url):
        env = {"DECIDER_DOMAINS": _domains(general=g_url), "GATEWAY_API_KEY": _API_KEY}
        with _gateway(env) as gw:
            denied, _h, _raw = _post(gw)
            wrong, _h2, _raw2 = _post(gw, headers={"Authorization": "Bearer nope"})
            allowed, _h3, raw = _post(gw, headers={"Authorization": f"Bearer {_API_KEY}"})
    assert (denied, wrong) == (401, 401)
    assert allowed == 200
    assert raw == _ANSWER
    assert len(general.seen) == 1


# ===========================================================================
# The pure seam and the config contract
# ===========================================================================


def test_decider_domains_parse_like_hand_lora_modules():
    table, _cfg = build_config(
        {
            "PRIMARY_SERVED_NAME": "P",
            "DECIDER_DOMAINS": (
                " general=http://decider-general:8000/ , ops=http://decider-ops:8000,"
                "broken,=http://x:1,Bad=http://y:1,empty=, ops=http://dup:1"
            ),
        }
    )
    assert dict(table.decider_domains) == {
        "general": "http://decider-general:8000",
        "ops": "http://decider-ops:8000",
    }


def test_decider_domains_empty_with_base_url_is_general():
    table, _cfg = build_config(
        {"PRIMARY_SERVED_NAME": "P", "DECIDER_BASE_URL": "http://decider:8000/"}
    )
    assert dict(table.decider_domains) == {"general": "http://decider:8000"}


def test_the_decider_backend_is_path_routed_never_model_routed():
    """The decider's id must not be a chat model nor be listed on /v1/models."""
    table, _cfg = build_config(
        {"PRIMARY_SERVED_NAME": "P", "DECIDER_DOMAINS": "general=http://decider-general:8000"}
    )
    decider = next(b for b in table.backends if b.name == "decider")
    assert decider.task == DECIDE_TASK
    assert decider.base_url == "http://decider-general:8000"
    assert decider not in model_routed_backends(table)
    ids = [m["id"] for m in list_models_payload(table)["data"]]
    assert decider.served_name not in ids
    assert "decider" not in table.infeasible


def test_resolve_decider_route_pure_seam():
    table, _cfg = build_config(
        {"PRIMARY_SERVED_NAME": "P", "DECIDER_DOMAINS": "general=http://g:1,ops=http://o:1"}
    )
    general = resolve_decider_route(table, None)
    assert (general.kind, general.domain, general.base_url) == ("local", "general", "http://g:1")
    assert general.requested == "decider:general"
    assert resolve_decider_route(table, "decider").domain == "general"
    ops = resolve_decider_route(table, " decider:ops ")
    assert (ops.kind, ops.domain, ops.base_url) == ("local", "ops", "http://o:1")
    assert resolve_decider_route(table, "decider:legal").kind == "model_not_found"
    unwired, _cfg = build_config({"PRIMARY_SERVED_NAME": "P"})
    assert resolve_decider_route(unwired, None).kind == "role_infeasible"


def test_is_systemone_path():
    assert is_systemone_path("/v1/systemone")
    assert is_systemone_path("/v1/systemone?x=1")
    assert not is_systemone_path("/v1/systemones")
    assert not is_systemone_path("/v1/chat/completions")
