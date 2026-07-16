"""Tests for the CLI's outbound gateway auth (issue #127, plan
``proxy-lobes-pairwise-auth``, task t3).

The fleet gateway is gaining OPT-IN inbound auth (a parallel task, #127 t2):
when ``GATEWAY_API_KEY`` (or its fallback ``CULTURE_VLLM_API_KEY``) is set in
the deployment's ``.env``, the gateway 401s a data-plane request missing or
mismatching ``Authorization: Bearer <key>``. This module covers the CLI-side
half: the CLI must attach that key automatically (so turning auth on doesn't
break the box's own tooling), and a wrong key must surface as one clear line,
never a Python traceback.

Three layers, matching the deliverable:

* :func:`lobes.cli._runtime_ops.gateway_auth_headers` — the pure resolution
  helper (precedence + never-raises contract), unit tested directly.
* :func:`lobes.assess.auth_headers` + ``_post``/``_get`` — the low-level
  attach mechanism, exercised against a real loopback server.
* ``lobes capabilities`` / ``lobes assess`` end to end via :func:`lobes.cli.main`
  against a real, auth-gated loopback server standing in for the gateway —
  the acceptance-criteria shape: succeeds with the right key, a wrong key
  gives a clear message (not a traceback), a keyless deployment sends no
  ``Authorization`` header at all (byte-identical to today).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import lobes.assess as A
from lobes.cli import _runtime_ops, main
from lobes.cli._commands import capabilities as capabilities_module
from lobes.roles import ROLES
from lobes.runtime import _compose, _env

_VALID_KEY = "s3cr3t-key"
_WRONG_KEY = "totally-wrong"

# Captured at import time — BEFORE tests/conftest.py's autouse
# ``offline_runtime`` fixture neutralises ``_fetch_gateway_capabilities`` for
# every other test in the suite. Mirrors ``tests/test_cli_capabilities.py``.
_REAL_FETCH_GATEWAY_CAPABILITIES = capabilities_module._fetch_gateway_capabilities


def _scaffold_fleet(path):
    """Write the packaged fleet templates verbatim (same as `lobes init --fleet`)."""
    _compose.write_scaffold(path, force=True, templates=_compose.FLEET_TEMPLATES)
    return path


def _capabilities_payload() -> dict:
    payload: dict[str, dict] = {}
    for i, role in enumerate(ROLES):
        payload[role] = {
            "role": role,
            "model": f"fake/{role}-model",
            "runtime": "vllm",
            "endpoint": "http://localhost:9999",
            "path": "/v1/fake",
            "context": 1000 + i,
            "quant": "fake-quant",
            "mtp": False,
            "feasible": True,
            "responsibilities": [f"{role}-thing"],
            "forbidden_responsibilities": [],
            "ready": True,
            "loaded": True,
        }
    return payload


class _AuthGatedHandler(BaseHTTPRequestHandler):
    """Stands in for an auth-enabled gateway: 401s unless the exact key is
    presented as ``Authorization: Bearer <key>``. Records every request's
    method/path/Authorization header on the class so tests can assert on
    what was (or wasn't) actually sent."""

    required_key: str = _VALID_KEY
    capabilities_payload: dict = {}
    seen: list = []

    def _record(self) -> None:
        type(self).seen.append(
            {
                "method": self.command,
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
            }
        )

    def _authorized(self) -> bool:
        return self.headers.get("Authorization") == f"Bearer {self.required_key}"

    def _write_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _unauthorized(self) -> None:
        self._write_json(
            401,
            {
                "error": {
                    "message": "invalid api key",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
        )

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._record()
        if not self._authorized():
            self._unauthorized()
            return
        if self.path == "/capabilities":
            self._write_json(200, self.capabilities_payload)
        elif self.path == "/health":
            self._write_json(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._write_json(200, {"data": [{"id": "foo/bar", "max_model_len": 32768}]})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._record()
        if not self._authorized():
            self._unauthorized()
            return
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = {}
        prompt = ""
        try:
            prompt = body["messages"][0]["content"]
        except (KeyError, IndexError, TypeError):
            pass
        if "17 * 23" in prompt:
            content = "= 391"
        elif "train" in prompt:
            content = "145 minutes"
        else:
            content = "ok"
        self._write_json(
            200,
            {
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 5, "prompt_tokens": 5},
            },
        )

    def log_message(self, *_a) -> None:  # silence test noise
        pass


@pytest.fixture
def auth_gateway(monkeypatch):
    """A real, auth-gated loopback server standing in for the gateway.

    Restores the REAL ``_fetch_gateway_capabilities`` (the autouse
    ``offline_runtime`` fixture stubs it to ``None`` by default) so
    ``lobes capabilities`` performs a genuine HTTP round trip, exactly like
    ``tests/test_cli_capabilities.py``'s ``fake_gateway`` fixture.
    """
    handler = type(
        "_BoundAuthGatedHandler",
        (_AuthGatedHandler,),
        {"required_key": _VALID_KEY, "capabilities_payload": _capabilities_payload(), "seen": []},
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(
        capabilities_module, "_fetch_gateway_capabilities", _REAL_FETCH_GATEWAY_CAPABILITIES
    )
    try:
        yield httpd.server_address[1], handler
    finally:
        httpd.shutdown()
        httpd.server_close()


def _deploy_dir_with_key(tmp_path, port: int, *, key: str | None, key_var: str = "GATEWAY_API_KEY"):
    _scaffold_fleet(tmp_path)
    _env.set_env(tmp_path / _compose.ENV_FILE, "VLLM_PORT", str(port))
    if key is not None:
        _env.set_env(tmp_path / _compose.ENV_FILE, key_var, key)
    return tmp_path


# ---------------------------------------------------------------------------
# 1. gateway_auth_headers — resolution precedence, never raises
# ---------------------------------------------------------------------------


def test_gateway_auth_headers_none_deploy_dir_is_noop() -> None:
    assert _runtime_ops.gateway_auth_headers(None) == {}


def test_gateway_auth_headers_unscaffolded_dir_is_noop(tmp_path) -> None:
    # No .env at all in this dir.
    assert _runtime_ops.gateway_auth_headers(tmp_path) == {}


def test_gateway_auth_headers_keyless_env_is_noop(tmp_path) -> None:
    _scaffold_fleet(tmp_path)
    assert _runtime_ops.gateway_auth_headers(tmp_path) == {}


def test_gateway_auth_headers_gateway_api_key_wins(tmp_path) -> None:
    _scaffold_fleet(tmp_path)
    _env.set_env(tmp_path / _compose.ENV_FILE, "GATEWAY_API_KEY", "gw-key")
    _env.set_env(tmp_path / _compose.ENV_FILE, "CULTURE_VLLM_API_KEY", "culture-key")
    assert _runtime_ops.gateway_auth_headers(tmp_path) == {"Authorization": "Bearer gw-key"}


def test_gateway_auth_headers_culture_fallback(tmp_path) -> None:
    _scaffold_fleet(tmp_path)
    _env.set_env(tmp_path / _compose.ENV_FILE, "CULTURE_VLLM_API_KEY", "culture-key")
    assert _runtime_ops.gateway_auth_headers(tmp_path) == {"Authorization": "Bearer culture-key"}


def test_gateway_auth_headers_blank_gateway_key_falls_back(tmp_path) -> None:
    _scaffold_fleet(tmp_path)
    _env.set_env(tmp_path / _compose.ENV_FILE, "GATEWAY_API_KEY", "")
    _env.set_env(tmp_path / _compose.ENV_FILE, "CULTURE_VLLM_API_KEY", "culture-key")
    assert _runtime_ops.gateway_auth_headers(tmp_path) == {"Authorization": "Bearer culture-key"}


def test_gateway_auth_headers_both_absent_empty(tmp_path) -> None:
    _scaffold_fleet(tmp_path)
    assert _runtime_ops.gateway_auth_headers(tmp_path) == {}


# ---------------------------------------------------------------------------
# 2. lobes.assess.auth_headers — the low-level _post/_get attach mechanism
# ---------------------------------------------------------------------------


def test_assess_post_get_send_no_header_by_default() -> None:
    """Outside any `with auth_headers(...)` block, _post/_get send nothing new."""
    handler = type(
        "_H", (_AuthGatedHandler,), {"required_key": "", "capabilities_payload": {}, "seen": []}
    )
    # required_key="" would never match "Bearer " unless Authorization is
    # also absent — use a permissive handler instead: record only.
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        # No `with A.auth_headers(...)` around this call.
        A._get(url, "/health")
    except Exception:
        pass  # the handler 401s without a matching key; only headers matter here
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert handler.seen, "request never reached the server"
    assert handler.seen[-1]["authorization"] is None


def test_assess_auth_headers_context_manager_attaches_header() -> None:
    handler = type(
        "_H2",
        (_AuthGatedHandler,),
        {"required_key": _VALID_KEY, "capabilities_payload": {}, "seen": []},
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        with A.auth_headers({"Authorization": f"Bearer {_VALID_KEY}"}):
            status, _ = A._get(url, "/health")
        assert status == 200
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert handler.seen[-1]["authorization"] == f"Bearer {_VALID_KEY}"


def test_assess_auth_headers_context_manager_resets_after_block() -> None:
    """Headers set inside the `with` block must not leak to calls made after it."""
    handler = type(
        "_H3",
        (_AuthGatedHandler,),
        {"required_key": _VALID_KEY, "capabilities_payload": {}, "seen": []},
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        with A.auth_headers({"Authorization": f"Bearer {_VALID_KEY}"}):
            A._get(url, "/health")
        # Outside the block: no header attached, so this 401s.
        with pytest.raises(Exception):
            status, _ = A._get(url, "/health")
            assert status == 401  # some urllib versions raise HTTPError instead
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert handler.seen[-1]["authorization"] is None


# ---------------------------------------------------------------------------
# 3a. `lobes capabilities` end to end (acceptance criterion a)
# ---------------------------------------------------------------------------


def test_capabilities_succeeds_with_correct_key_from_env(tmp_path, auth_gateway, capsys) -> None:
    port, handler = auth_gateway
    _deploy_dir_with_key(tmp_path, port, key=_VALID_KEY)
    rc = main(["capabilities", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == handler.capabilities_payload
    sent = [r for r in handler.seen if r["path"] == "/capabilities"]
    assert sent, "capabilities request never reached the mock gateway"
    assert sent[-1]["authorization"] == f"Bearer {_VALID_KEY}"


def test_capabilities_wrong_key_gives_clear_message_not_traceback(
    tmp_path, auth_gateway, capsys
) -> None:
    port, _handler = auth_gateway
    _deploy_dir_with_key(tmp_path, port, key=_WRONG_KEY)
    rc = main(["capabilities", "--compose-dir", str(tmp_path)])
    err = capsys.readouterr().err
    assert rc != 0
    assert "Traceback" not in err
    assert "gateway rejected the API key" in err
    assert "GATEWAY_API_KEY" in err
    assert "CULTURE_VLLM_API_KEY" in err
    assert str(tmp_path) in err  # names the actual deployment .env


def test_capabilities_missing_key_gives_clear_message(tmp_path, auth_gateway, capsys) -> None:
    """No key configured at all in .env while the gateway requires one."""
    port, _handler = auth_gateway
    _deploy_dir_with_key(tmp_path, port, key=None)
    rc = main(["capabilities", "--compose-dir", str(tmp_path)])
    err = capsys.readouterr().err
    assert rc != 0
    assert "Traceback" not in err
    assert "gateway rejected the API key" in err


# ---------------------------------------------------------------------------
# 3b. `lobes assess` end to end (acceptance criterion a)
# ---------------------------------------------------------------------------


def test_assess_succeeds_with_correct_key_from_env(tmp_path, auth_gateway, capsys) -> None:
    port, handler = auth_gateway
    _deploy_dir_with_key(tmp_path, port, key=_VALID_KEY)
    rc = main(["assess", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    posts = [r for r in handler.seen if r["method"] == "POST"]
    assert posts, "no chat-completions request reached the mock gateway"
    assert all(r["authorization"] == f"Bearer {_VALID_KEY}" for r in posts)


def test_assess_wrong_key_gives_clear_message_not_traceback(tmp_path, auth_gateway, capsys) -> None:
    port, _handler = auth_gateway
    _deploy_dir_with_key(tmp_path, port, key=_WRONG_KEY)
    rc = main(["assess", "--compose-dir", str(tmp_path)])
    err = capsys.readouterr().err
    assert rc != 0
    assert "Traceback" not in err
    assert "gateway rejected the API key" in err
    assert "GATEWAY_API_KEY" in err
    assert "CULTURE_VLLM_API_KEY" in err


def test_assess_wrong_key_json_mode_clear_message(tmp_path, auth_gateway, capsys) -> None:
    port, _handler = auth_gateway
    _deploy_dir_with_key(tmp_path, port, key=_WRONG_KEY)
    rc = main(["assess", "--compose-dir", str(tmp_path), "--json"])
    out, err = capsys.readouterr()
    assert rc != 0
    assert out == ""  # a failure never writes a partial/malformed result to stdout
    payload = json.loads(err)  # ModelGearError is always emitted as JSON to stderr
    assert "Traceback" not in payload["message"]
    assert "gateway rejected the API key" in payload["message"]


def test_assess_culture_vllm_api_key_fallback_also_works(tmp_path, auth_gateway, capsys) -> None:
    """CULTURE_VLLM_API_KEY (no GATEWAY_API_KEY set) also authenticates."""
    port, handler = auth_gateway
    _deploy_dir_with_key(tmp_path, port, key=_VALID_KEY, key_var="CULTURE_VLLM_API_KEY")
    rc = main(["assess", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    posts = [r for r in handler.seen if r["method"] == "POST"]
    assert all(r["authorization"] == f"Bearer {_VALID_KEY}" for r in posts)


# ---------------------------------------------------------------------------
# 3c. Keyless deployment — byte-identical, NO Authorization header at all
#     (acceptance criterion b)
# ---------------------------------------------------------------------------


class _RecordingOpenHandler(BaseHTTPRequestHandler):
    """A permissive stand-in gateway that answers everything 200 regardless of
    auth, but records every request's headers — used to prove a keyless
    deployment sends NO Authorization header at all (not even a blank one)."""

    seen: list = []

    def _record(self) -> None:
        type(self).seen.append(
            {"method": self.command, "path": self.path, "headers": dict(self.headers.items())}
        )

    def _write_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        self._record()
        if self.path == "/capabilities":
            self._write_json(200, _capabilities_payload())
        elif self.path == "/health":
            self._write_json(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._write_json(200, {"data": [{"id": "foo/bar", "max_model_len": 32768}]})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        self._record()
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length:
            self.rfile.read(length)
        self._write_json(
            200,
            {
                "choices": [{"message": {"content": "= 391"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 5, "prompt_tokens": 5},
            },
        )

    def log_message(self, *_a) -> None:
        pass


@pytest.fixture
def open_gateway(monkeypatch):
    handler = type("_BoundOpenHandler", (_RecordingOpenHandler,), {"seen": []})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(
        capabilities_module, "_fetch_gateway_capabilities", _REAL_FETCH_GATEWAY_CAPABILITIES
    )
    try:
        yield httpd.server_address[1], handler
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_keyless_deployment_capabilities_sends_no_authorization_header(
    tmp_path, open_gateway, capsys
) -> None:
    port, handler = open_gateway
    _scaffold_fleet(tmp_path)
    _env.set_env(tmp_path / _compose.ENV_FILE, "VLLM_PORT", str(port))
    # No GATEWAY_API_KEY / CULTURE_VLLM_API_KEY set anywhere — the default,
    # untouched fleet scaffold.
    rc = main(["capabilities", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    capsys.readouterr()
    sent = [r for r in handler.seen if r["path"] == "/capabilities"]
    assert sent
    assert "Authorization" not in sent[-1]["headers"]


def test_keyless_deployment_assess_sends_no_authorization_header(
    tmp_path, open_gateway, capsys
) -> None:
    port, handler = open_gateway
    _scaffold_fleet(tmp_path)
    _env.set_env(tmp_path / _compose.ENV_FILE, "VLLM_PORT", str(port))
    rc = main(["assess", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    capsys.readouterr()
    assert handler.seen
    for record in handler.seen:
        assert "Authorization" not in record["headers"]


def test_keyless_gateway_auth_headers_helper_returns_empty_dict(tmp_path) -> None:
    """Direct check on the shared helper: byte-identical (no key) means `{}`,
    never e.g. `{"Authorization": "Bearer None"}` or similar."""
    _scaffold_fleet(tmp_path)
    headers = _runtime_ops.gateway_auth_headers(tmp_path)
    assert headers == {}
    assert "Authorization" not in headers


# ---------------------------------------------------------------------------
# 4. `lobes measure` also attaches the header — WITHOUT lobes/roles_measure.py
#    being touched at all. `lobes.roles_measure` calls `lobes.assess._post` /
#    `measure_prefill_ttft` directly by name for the cortex/senses probes, so
#    `lobes.assess.auth_headers()`'s contextvar mechanism reaches them too —
#    this is the "single with-block at each CLI dispatch boundary covers
#    every read-only verb" claim made in lobes/assess.py's module comment,
#    demonstrated end to end. /health and /metrics stay unauthenticated here
#    (matching lobes._metrics's documented "vLLM serves /metrics and /health
#    unauthenticated" contract) — only the data-plane chat-completions probe
#    is auth-gated.
# ---------------------------------------------------------------------------


class _MeasureAuthHandler(BaseHTTPRequestHandler):
    required_key: str = _VALID_KEY
    seen: list = []

    def _write_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/health", "/metrics"):
            self._write_json(200, {"status": "ok"})
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        type(self).seen.append(
            {"path": self.path, "authorization": self.headers.get("Authorization")}
        )
        if self.headers.get("Authorization") != f"Bearer {self.required_key}":
            self._write_json(401, {"error": {"message": "invalid api key"}})
            return
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length:
            self.rfile.read(length)
        self._write_json(
            200,
            {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 4, "prompt_tokens": 4},
            },
        )

    def log_message(self, *_a) -> None:
        pass


def test_measure_cortex_attaches_key_from_env_via_roles_measure(tmp_path, capsys) -> None:
    """`lobes measure --role cortex` reuses `lobes.assess._post`/
    `measure_prefill_ttft` inside `lobes.roles_measure` — proving the
    contextvar-based `auth_headers()` mechanism reaches that module's probes
    without `lobes/roles_measure.py` itself needing any change."""
    handler = type("_BoundMeasureAuthHandler", (_MeasureAuthHandler,), {"seen": []})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        port = httpd.server_address[1]
        _deploy_dir_with_key(tmp_path, port, key=_VALID_KEY)
        rc = main(["measure", "--role", "cortex", "--compose-dir", str(tmp_path), "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["cortex"]["ready"] is True
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert handler.seen, "no chat-completions request reached the mock gateway"
    assert all(r["authorization"] == f"Bearer {_VALID_KEY}" for r in handler.seen)
