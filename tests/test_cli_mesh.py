"""Tests for ``lobes mesh status|request|approve|revoke`` (mesh-brain-join,
task t10) plus the additive ``lobes capabilities`` member/suffixed-lane
rendering (c47, h38).

``lobes mesh`` is registered as a noun exactly like ``lobes fleet``
(``lobes/cli/_commands/fleet.py``): a bare ``lobes mesh`` is the read-only
``status`` default. ``request``/``approve``/``revoke`` are write verbs —
dry-run by default (prints the plan, changes nothing), ``--apply`` commits.

Network calls are never left to hit whatever happens to be listening on the
guessed port on the dev rig (see ``tests/conftest.py``'s own warning about
that): every test either stays on the dry-run path (no I/O at all) or spins
a real ``ThreadingHTTPServer`` on an ephemeral port and passes ``--port``
explicitly, mirroring ``tests/test_cli_capabilities.py``'s ``fake_gateway``
fixture.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from lobes.cli import main
from lobes.cli._commands import capabilities as capabilities_module
from lobes.roles import ROLES
from lobes.runtime import _compose, _env

# Captured at import time — BEFORE tests/conftest.py's autouse
# ``offline_runtime`` fixture neutralises ``_fetch_gateway_capabilities`` for
# every other test in the suite (same pattern as
# ``tests/test_cli_capabilities.py``).
_REAL_FETCH_GATEWAY_CAPABILITIES = capabilities_module._fetch_gateway_capabilities

# ---------------------------------------------------------------------------
# shared fake-gateway harness
# ---------------------------------------------------------------------------


class _FakeMeshHandler(BaseHTTPRequestHandler):
    """Serves canned mesh responses; captures the last request seen."""

    roster_payload: dict = {"members": []}
    join_status: int = 202
    join_payload: dict = {"status": "pending join registered"}
    approve_status: int = 200
    approve_payload: dict = {"status": "approved"}
    revoke_status: int = 200
    revoke_payload: dict = {"status": "revoked"}
    reannounce_status: int = 200
    reannounce_payload: dict = {"status": "reannounced", "name": "me"}
    require_key: str | None = None
    last_request: dict | None = None

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return {}

    def _unauthorized(self) -> bool:
        if self.require_key is None:
            return False
        auth = self.headers.get("Authorization", "")
        return auth != f"Bearer {self.require_key}"

    def _send(self, status: int, payload) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/mesh/roster":
            if self._unauthorized():
                self._send(401, {"error": {"message": "Invalid API key."}})
                return
            self._send(200, self.roster_payload)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        body = self._read_body()
        type(self).last_request = {"path": self.path, "body": body, "headers": dict(self.headers)}
        if self.path == "/mesh/join":
            self._send(self.join_status, self.join_payload)
        elif self.path == "/mesh/approve":
            if self._unauthorized():
                self._send(401, {"error": {"message": "Invalid API key."}})
                return
            self._send(self.approve_status, self.approve_payload)
        elif self.path == "/mesh/revoke":
            if self._unauthorized():
                self._send(401, {"error": {"message": "Invalid API key."}})
                return
            self._send(self.revoke_status, self.revoke_payload)
        elif self.path == "/mesh/reannounce":
            if self._unauthorized():
                self._send(401, {"error": {"message": "Invalid API key."}})
                return
            self._send(self.reannounce_status, self.reannounce_payload)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *_a) -> None:  # silence test noise
        pass


@pytest.fixture
def fake_mesh():
    handler = type("_BoundFakeMeshHandler", (_FakeMeshHandler,), {})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_address[1], handler
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---------------------------------------------------------------------------
# AC-1: registration + approve dry-run vs --apply
# ---------------------------------------------------------------------------


def test_mesh_registered_like_fleet_noun() -> None:
    """'lobes mesh' resolves without argparse blowing up — same noun shape
    as 'lobes fleet' (bare noun -> read-only default)."""
    rc = main(["mesh", "--port", "1", "--json"])
    assert rc == 0  # unreachable port -> {"available": False, ...}, not a crash


def test_mesh_approve_without_apply_prints_plan_and_changes_nothing(fake_mesh, capsys) -> None:
    port, handler = fake_mesh
    rc = main(["mesh", "approve", "bob", "--port", str(port)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "bob" in out
    assert "--apply" in out
    # Nothing was ever sent to the fake gateway.
    assert handler.last_request is None


def test_mesh_approve_with_apply_posts_with_join_key(tmp_path, fake_mesh, capsys) -> None:
    port, handler = fake_mesh
    handler.require_key = "sk-mesh-test"
    _compose.write_scaffold(tmp_path, force=True)
    _env.set_env(tmp_path / _compose.ENV_FILE, "LOBES_MESH_KEY", "sk-mesh-test")
    rc = main(
        [
            "mesh",
            "approve",
            "bob",
            "--for",
            "24h",
            "--compose-dir",
            str(tmp_path),
            "--port",
            str(port),
            "--apply",
        ]
    )
    assert rc == 0
    assert handler.last_request is not None
    assert handler.last_request["path"] == "/mesh/approve"
    assert handler.last_request["body"]["name"] == "bob"
    assert handler.last_request["body"]["expiry"] == pytest.approx(86400.0)
    assert handler.last_request["headers"]["Authorization"] == "Bearer sk-mesh-test"
    out = capsys.readouterr().out
    assert "approved" in out.lower()


def test_mesh_approve_apply_without_key_refuses_before_any_request(tmp_path, fake_mesh) -> None:
    port, handler = fake_mesh
    _compose.write_scaffold(tmp_path, force=True)  # no LOBES_MESH_KEY set
    rc = main(
        [
            "mesh",
            "approve",
            "bob",
            "--compose-dir",
            str(tmp_path),
            "--port",
            str(port),
            "--apply",
        ]
    )
    assert rc != 0
    assert handler.last_request is None  # refused locally, never sent


def test_mesh_revoke_without_apply_is_a_dry_run(fake_mesh, capsys) -> None:
    port, handler = fake_mesh
    rc = main(["mesh", "revoke", "bob", "--port", str(port)])
    assert rc == 0
    assert "DRY RUN" in capsys.readouterr().out
    assert handler.last_request is None


def test_mesh_revoke_with_apply_posts_with_join_key(tmp_path, fake_mesh) -> None:
    port, handler = fake_mesh
    handler.require_key = "sk-mesh-test"
    _compose.write_scaffold(tmp_path, force=True)
    _env.set_env(tmp_path / _compose.ENV_FILE, "LOBES_MESH_KEY", "sk-mesh-test")
    rc = main(
        [
            "mesh",
            "revoke",
            "bob",
            "--compose-dir",
            str(tmp_path),
            "--port",
            str(port),
            "--apply",
        ]
    )
    assert rc == 0
    assert handler.last_request["path"] == "/mesh/revoke"
    assert handler.last_request["body"]["name"] == "bob"
    assert handler.last_request["headers"]["Authorization"] == "Bearer sk-mesh-test"


# ---------------------------------------------------------------------------
# t8 follow-up (c27/h1, c46/h37): trigger_reannounce — the switch/up hook
# ---------------------------------------------------------------------------


def test_trigger_reannounce_noop_when_key_unset(fake_mesh) -> None:
    from lobes.cli._commands.mesh import trigger_reannounce

    port, handler = fake_mesh
    trigger_reannounce(port, {})
    assert handler.last_request is None  # no request at all


def test_trigger_reannounce_posts_with_bearer_key_when_set(fake_mesh) -> None:
    from lobes.cli._commands.mesh import trigger_reannounce

    port, handler = fake_mesh
    handler.require_key = "sk-mesh-test"
    trigger_reannounce(port, {"LOBES_MESH_KEY": "sk-mesh-test"})
    assert handler.last_request is not None
    assert handler.last_request["path"] == "/mesh/reannounce"
    assert handler.last_request["headers"]["Authorization"] == "Bearer sk-mesh-test"


def test_trigger_reannounce_swallows_errors_on_unreachable_gateway() -> None:
    from lobes.cli._commands.mesh import trigger_reannounce

    # Nothing listens on port 1 — must not raise.
    trigger_reannounce(1, {"LOBES_MESH_KEY": "sk-mesh-test"})


def test_trigger_reannounce_swallows_a_non_2xx_response(fake_mesh) -> None:
    from lobes.cli._commands.mesh import trigger_reannounce

    port, handler = fake_mesh
    handler.require_key = "sk-mesh-test"
    # Wrong key -> the fake gateway 401s -> HTTPError -> swallowed, no raise.
    trigger_reannounce(port, {"LOBES_MESH_KEY": "wrong-key"})
    assert handler.last_request is not None
    assert handler.last_request["path"] == "/mesh/reannounce"


# ---------------------------------------------------------------------------
# t8 follow-up: lobes switch/up call trigger_reannounce on --apply, never
# on a dry run.
# ---------------------------------------------------------------------------


def test_switch_apply_triggers_reannounce_when_key_set(tmp_path, monkeypatch) -> None:
    from lobes.cli._commands import switch as switch_module

    _compose.write_scaffold(tmp_path, force=True)
    _env.set_env(tmp_path / _compose.ENV_FILE, "LOBES_MESH_KEY", "sk-mesh-test")

    monkeypatch.setattr(switch_module._runtime_ops, "compose_check", lambda *a, **k: None)
    monkeypatch.setattr(switch_module._health, "wait_health", lambda *a, **k: None)
    monkeypatch.setattr(switch_module._runtime_ops, "probe_tool_calling", lambda *a, **k: None)

    calls = []
    monkeypatch.setattr(
        switch_module, "trigger_reannounce", lambda port, env: calls.append((port, env))
    )

    rc = main(
        [
            "switch",
            "unsloth/Qwen3.8-27B-NVFP4",
            "--compose-dir",
            str(tmp_path),
            "--no-probe",
            "--apply",
        ]
    )
    assert rc == 0
    assert len(calls) == 1
    assert calls[0][1].get("LOBES_MESH_KEY") == "sk-mesh-test"


def test_switch_dry_run_never_triggers_reannounce(tmp_path, monkeypatch) -> None:
    from lobes.cli._commands import switch as switch_module

    _compose.write_scaffold(tmp_path, force=True)
    _env.set_env(tmp_path / _compose.ENV_FILE, "LOBES_MESH_KEY", "sk-mesh-test")

    calls = []
    monkeypatch.setattr(
        switch_module, "trigger_reannounce", lambda port, env: calls.append((port, env))
    )

    rc = main(
        [
            "switch",
            "unsloth/Qwen3.8-27B-NVFP4",
            "--compose-dir",
            str(tmp_path),
            "--no-probe",
        ]
    )
    assert rc == 0
    assert calls == []


def test_up_apply_triggers_reannounce_when_key_set(tmp_path, monkeypatch) -> None:
    from lobes.cli._commands import up as up_module

    _compose.write_scaffold(tmp_path, force=True)
    _env.set_env(tmp_path / _compose.ENV_FILE, "LOBES_MESH_KEY", "sk-mesh-test")

    monkeypatch.setattr(up_module._runtime_ops, "compose_check", lambda *a, **k: None)

    calls = []
    monkeypatch.setattr(
        up_module, "trigger_reannounce", lambda port, env: calls.append((port, env))
    )

    rc = main(["up", "cortex", "--compose-dir", str(tmp_path), "--apply"])
    assert rc == 0
    assert len(calls) == 1
    assert calls[0][1].get("LOBES_MESH_KEY") == "sk-mesh-test"


def test_up_dry_run_never_triggers_reannounce(tmp_path, monkeypatch) -> None:
    from lobes.cli._commands import up as up_module

    _compose.write_scaffold(tmp_path, force=True)
    _env.set_env(tmp_path / _compose.ENV_FILE, "LOBES_MESH_KEY", "sk-mesh-test")

    calls = []
    monkeypatch.setattr(
        up_module, "trigger_reannounce", lambda port, env: calls.append((port, env))
    )

    rc = main(["up", "cortex", "--compose-dir", str(tmp_path)])
    assert rc == 0
    assert calls == []


# ---------------------------------------------------------------------------
# AC-2: 'lobes mesh status' golden against a fake roster
# ---------------------------------------------------------------------------


_GOLDEN_ROSTER = {
    "members": [
        {
            "name": "spark",
            "origin": "http://spark.example:8000",
            "capacity": 2.0,
            "last_seen_age": 12,
            "expiry": 3599,
            "verified": True,
            "roles": ["cortex", "senses"],
        },
        {
            "name": "thor",
            "origin": "http://thor.example:8000",
            "capacity": 1.0,
            "last_seen_age": 301,
            "expiry": 0,
            "verified": False,
            "roles": ["worker"],
        },
        {
            "name": "orin",
            "origin": "http://orin.example:8000",
            "capacity": 1.0,
            "last_seen_age": 5,
            "expiry": 1800,
            "flapping": True,
            "roles": [],
        },
    ]
}


def test_mesh_status_golden_table_against_fake_roster(fake_mesh, capsys) -> None:
    port, handler = fake_mesh
    handler.roster_payload = _GOLDEN_ROSTER
    rc = main(["mesh", "status", "--port", str(port)])
    assert rc == 0
    out = capsys.readouterr().out

    # name
    assert "spark" in out
    assert "thor" in out
    assert "orin" in out
    # last-heartbeat age
    assert "12s" in out
    assert "301s" in out
    # approval expiry
    assert "3599s" in out
    assert "1800s" in out
    # verified/unverified/flapping
    assert "verified" in out
    assert "unverified" in out
    assert "flapping" in out
    # roles
    assert "cortex" in out
    assert "senses" in out
    assert "worker" in out


def test_mesh_status_golden_json_against_fake_roster(fake_mesh, capsys) -> None:
    port, handler = fake_mesh
    handler.roster_payload = _GOLDEN_ROSTER
    rc = main(["mesh", "status", "--port", str(port), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["available"] is True
    assert payload["members"] == _GOLDEN_ROSTER["members"]


def test_mesh_status_unreachable_gateway_renders_unavailable_not_a_crash(capsys) -> None:
    # Nothing listens on this port — must degrade cleanly, never traceback.
    rc = main(["mesh", "status", "--port", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "unavailable" in out.lower() or "disabled" in out.lower()


def test_mesh_status_json_unreachable_gateway(capsys) -> None:
    rc = main(["mesh", "status", "--port", "1", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"available": False, "members": []}


def test_mesh_status_minimal_pinned_roster_shape_still_renders(fake_mesh, capsys) -> None:
    """The REAL /mesh/roster route today returns only {name, origin,
    capacity} per member (see _mesh_routes.roster_list) — the richer golden
    above is an additive superset a future task may fill in. This proves the
    CLI never crashes on the minimal, currently-pinned shape."""
    port, handler = fake_mesh
    handler.roster_payload = {
        "members": [{"name": "spark", "origin": "http://spark.example:8000", "capacity": 2.0}]
    }
    rc = main(["mesh", "status", "--port", str(port)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "spark" in out
    assert "unknown" in out  # verified/unverified/flapping falls back honestly


# ---------------------------------------------------------------------------
# AC-2 (second half): 'lobes mesh request' works with no join key set
# ---------------------------------------------------------------------------


def test_mesh_request_without_apply_is_dry_run_and_needs_no_key(
    tmp_path, fake_mesh, capsys
) -> None:
    port, handler = fake_mesh
    _compose.write_scaffold(tmp_path, force=True)  # no LOBES_MESH_KEY anywhere
    rc = main(
        [
            "mesh",
            "request",
            "--name",
            "newbox",
            "--origin",
            "http://newbox.example:8000",
            "--compose-dir",
            str(tmp_path),
            "--port",
            str(port),
        ]
    )
    assert rc == 0
    assert "DRY RUN" in capsys.readouterr().out
    assert handler.last_request is None


def test_mesh_request_with_apply_posts_keyless_with_no_join_key_configured(
    tmp_path, fake_mesh, capsys
) -> None:
    port, handler = fake_mesh
    _compose.write_scaffold(tmp_path, force=True)  # LOBES_MESH_KEY deliberately unset
    rc = main(
        [
            "mesh",
            "request",
            "--name",
            "newbox",
            "--origin",
            "http://newbox.example:8000",
            "--compose-dir",
            str(tmp_path),
            "--port",
            str(port),
            "--apply",
        ]
    )
    assert rc == 0
    assert handler.last_request is not None
    assert handler.last_request["path"] == "/mesh/join"
    assert handler.last_request["body"] == {
        "name": "newbox",
        "origin": "http://newbox.example:8000",
        "capacity": None,
    }
    # No Authorization header at all — the join endpoint is keyless.
    assert "Authorization" not in handler.last_request["headers"]
    out = capsys.readouterr().out
    assert "requested join" in out.lower()


def test_mesh_request_defaults_name_and_origin_from_env(tmp_path, fake_mesh) -> None:
    port, handler = fake_mesh
    _compose.write_scaffold(tmp_path, force=True)
    _env.set_env(tmp_path / _compose.ENV_FILE, "LOBES_MESH_NAME", "envbox")
    _env.set_env(tmp_path / _compose.ENV_FILE, "GATEWAY_SELF_ORIGIN", "http://envbox.example:8000")
    rc = main(["mesh", "request", "--compose-dir", str(tmp_path), "--port", str(port), "--apply"])
    assert rc == 0
    assert handler.last_request["body"]["name"] == "envbox"
    assert handler.last_request["body"]["origin"] == "http://envbox.example:8000"


def test_mesh_request_apply_without_any_name_source_refuses(tmp_path, fake_mesh) -> None:
    port, handler = fake_mesh
    _compose.write_scaffold(tmp_path, force=True)
    rc = main(["mesh", "request", "--compose-dir", str(tmp_path), "--port", str(port), "--apply"])
    assert rc != 0
    assert handler.last_request is None


def test_mesh_request_http_error_surfaces_as_friendly_error(tmp_path, fake_mesh) -> None:
    port, handler = fake_mesh
    handler.join_status = 400
    handler.join_payload = {"error": "same origin already pending"}
    _compose.write_scaffold(tmp_path, force=True)
    rc = main(
        [
            "mesh",
            "request",
            "--name",
            "dup",
            "--origin",
            "http://dup.example:8000",
            "--compose-dir",
            str(tmp_path),
            "--port",
            str(port),
            "--apply",
        ]
    )
    assert rc != 0


# ---------------------------------------------------------------------------
# AC-3: lobes capabilities shows each role's serving member and suffixed lanes
# ---------------------------------------------------------------------------


class _FakeCapabilitiesHandler(BaseHTTPRequestHandler):
    payload: dict = {}

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/capabilities":
            body = json.dumps(self.payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *_a) -> None:
        pass


def _capabilities_payload_with_mesh_fields() -> dict:
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
            "member": "spark" if role == "cortex" else None,
            "suffixed_lanes": ["cortex-thor", "cortex-orin"] if role == "cortex" else [],
        }
    return payload


def test_capabilities_table_shows_serving_member_and_suffixed_lanes(monkeypatch, capsys) -> None:
    payload = _capabilities_payload_with_mesh_fields()
    handler = type(
        "_BoundFakeCapabilitiesHandler", (_FakeCapabilitiesHandler,), {"payload": payload}
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    # Restore the real fetch (tests/conftest.py's autouse fixture stubs it to
    # None) so this hits the real HTTP round trip against the fake server.
    monkeypatch.setattr(
        capabilities_module, "_fetch_gateway_capabilities", _REAL_FETCH_GATEWAY_CAPABILITIES
    )
    try:
        rc = main(["capabilities", "--port", str(httpd.server_address[1])])
        assert rc == 0
        out = capsys.readouterr().out
        assert "served by mesh member: spark" in out
        assert "suffixed lanes: cortex-thor, cortex-orin" in out
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_capabilities_offline_fallback_unaffected_by_mesh_rendering(tmp_path, capsys) -> None:
    """Offline fallback (no gateway reachable) still renders with ready=False
    and never fabricates a member/suffixed-lane line."""
    _compose.write_scaffold(tmp_path, force=True)
    rc = main(["capabilities", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    out, err = capsys.readouterr()
    payload = json.loads(out)
    assert "offline" in err
    for role in ROLES:
        assert payload[role]["ready"] is False

    rc = main(["capabilities", "--compose-dir", str(tmp_path)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "served by mesh member:" not in text
    assert "suffixed lanes:" not in text


# ---------------------------------------------------------------------------
# review #252 finding 16: --for must reject nan/inf, not just <= 0
# ---------------------------------------------------------------------------


class TestParseDurationFiniteness:
    def test_rejects_nan(self) -> None:
        from lobes.cli._commands.mesh import _parse_duration_seconds
        from lobes.cli._errors import ModelGearError

        with pytest.raises(ModelGearError):
            _parse_duration_seconds("nan")

    def test_rejects_inf(self) -> None:
        from lobes.cli._commands.mesh import _parse_duration_seconds
        from lobes.cli._errors import ModelGearError

        with pytest.raises(ModelGearError):
            _parse_duration_seconds("inf")

    def test_rejects_negative_inf(self) -> None:
        from lobes.cli._commands.mesh import _parse_duration_seconds
        from lobes.cli._errors import ModelGearError

        with pytest.raises(ModelGearError):
            _parse_duration_seconds("-inf")

    def test_accepts_ordinary_duration(self) -> None:
        from lobes.cli._commands.mesh import _parse_duration_seconds

        assert _parse_duration_seconds("24h") == pytest.approx(86400.0)


# ---------------------------------------------------------------------------
# mesh-boot-window-and-capabilities-advert (task t5, c23/h19): a member
# record that has never yet been probed carries `"probed": false` and
# `unverified_reason: "not_yet_probed"` — the roster table already prints
# any `unverified_reason` via `.get` (Item C, t9); this pins the additive
# `(not probed)` marker on the three-state status label alongside it.
# ---------------------------------------------------------------------------

_UNPROBED_ROSTER = {
    "members": [
        {
            "name": "thor",
            "origin": "http://thor.example:8000",
            "capacity": 1.0,
            "last_seen_age": 3,
            "expiry": 3600,
            "verified": False,
            "probed": False,
            "unverified_reason": "not_yet_probed",
            "roles": ["worker"],
        },
    ]
}


def test_mesh_status_marks_unprobed_member_with_reason_and_status_marker(fake_mesh, capsys) -> None:
    port, handler = fake_mesh
    handler.roster_payload = _UNPROBED_ROSTER
    rc = main(["mesh", "status", "--port", str(port)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "unverified_reason: not_yet_probed" in out
    assert "unverified (not probed)" in out


def test_mesh_status_probed_member_has_no_unprobed_marker(fake_mesh, capsys) -> None:
    """A clean probe clears both `probed` (True) and `unverified_reason`
    (None) — the roster table must not fabricate a marker that isn't in the
    payload."""
    port, handler = fake_mesh
    handler.roster_payload = {
        "members": [
            {
                "name": "thor",
                "origin": "http://thor.example:8000",
                "capacity": 1.0,
                "verified": True,
                "probed": True,
                "unverified_reason": None,
                "roles": ["worker"],
            },
        ]
    }
    rc = main(["mesh", "status", "--port", str(port)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "(not probed)" not in out
    assert "unverified_reason" not in out
