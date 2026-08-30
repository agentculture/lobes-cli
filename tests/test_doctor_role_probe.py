"""`lobes doctor --role` — the live per-role probe (issue #234).

Offline throughout: every check is driven through an injected request function,
so the suite exercises the DECISIONS without opening a socket.
"""

from __future__ import annotations

import lobes.cli._commands._role_probe as rp


def _entry(**kw) -> dict:
    base = {
        "role": "associate",
        "model": "nvidia/Some-Checkpoint-NVFP4",
        "context": 128000,
        "ready": True,
        "feasible": True,
        "proxied": None,
        "hosted_by": None,
    }
    base.update(kw)
    return base


def _ok_completion(text: str = "ok") -> dict:
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}


def _ids(checks) -> dict[str, dict]:
    return {c["id"]: c for c in checks}


def test_a_missing_advert_names_a_stale_gateway_rather_than_a_dead_role() -> None:
    checks = _ids(rp.advert_check(None, "associate"))
    assert checks["role_advert"]["passed"] is False
    assert "lobes up gateway" in checks["role_advert"]["remediation"]


def test_a_proxied_role_is_not_reported_broken_for_feasible_false() -> None:
    """`feasible:false` on a proxied role means "not hosted here", not "faulty".

    Misreading that flag is half of #234's ask 2, so the probe says it plainly.
    """
    checks = _ids(
        rp.advert_check(
            _entry(feasible=False, proxied=True, hosted_by="http://peer:8000", ready=True),
            "associate",
        )
    )
    assert checks["role_proxied"]["passed"] is True
    assert "proxied to http://peer:8000" in checks["role_advert"]["message"]


def test_a_window_disagreement_is_an_error_not_a_note(monkeypatch) -> None:
    """The advert saying 1048576 while the engine serves 128000 is THE defect."""
    monkeypatch.setattr(rp, "_request", lambda *a, **k: (200, {"max_model_len": 128000}, 0.01))
    check = rp.window_check("http://gw", "associate", 1048576, {}, 5.0)
    assert check["passed"] is False
    assert check["severity"] == "error"
    assert "1048576" in check["message"]
    assert "128000" in check["message"]


def test_a_matching_window_passes(monkeypatch) -> None:
    monkeypatch.setattr(rp, "_request", lambda *a, **k: (200, {"max_model_len": 128000}, 0.01))
    assert rp.window_check("http://gw", "associate", 128000, {}, 5.0)["passed"] is True


def test_an_empty_200_is_a_failure_not_a_healthy_lane(monkeypatch) -> None:
    """The measured caller hazard: HTTP 200 carrying no content looks like success."""
    monkeypatch.setattr(rp, "_request", lambda *a, **k: (200, _ok_completion(""), 0.4))
    check = rp.alias_check("http://gw", "associate", {}, 5.0)
    assert check["passed"] is False
    assert check["severity"] == "error"
    assert "EMPTY" in check["message"]
    assert "enable_thinking" in check["remediation"]


def test_a_refused_alias_is_an_error(monkeypatch) -> None:
    body = {"error": {"code": "role_infeasible"}}
    monkeypatch.setattr(rp, "_request", lambda *a, **k: (404, body, 0.02))
    check = rp.alias_check("http://gw", "associate", {}, 5.0)
    assert check["passed"] is False
    assert "role_infeasible" in check["message"]


def test_a_working_alias_passes(monkeypatch) -> None:
    monkeypatch.setattr(rp, "_request", lambda *a, **k: (200, _ok_completion(), 0.19))
    check = rp.alias_check("http://gw", "associate", {}, 5.0)
    assert check["passed"] is True
    assert "0.19s" in check["message"]


def test_an_unroutable_served_id_names_the_alias_to_use(monkeypatch) -> None:
    """#234 ask 3: a consumer must be able to recover deterministically."""
    body = {"error": {"code": "role_infeasible"}}
    monkeypatch.setattr(rp, "_request", lambda *a, **k: (404, body, 0.02))
    check = rp.served_id_check("http://gw", "associate", "nvidia/X", {}, 5.0)
    assert check is not None
    assert "address this role as `associate`" in check["message"]
    assert "model=associate" in check["remediation"]


def test_the_served_id_check_is_skipped_when_it_is_the_alias() -> None:
    assert rp.served_id_check("http://gw", "associate", "associate", {}, 5.0) is None
    assert rp.served_id_check("http://gw", "associate", None, {}, 5.0) is None


def test_a_connection_failure_never_raises(monkeypatch) -> None:
    """Status 0 distinguishes "nothing answered" from "answered badly"."""
    monkeypatch.setattr(rp, "_request", lambda *a, **k: (0, None, 0.0))
    checks = _ids(rp.probe_role("http://gw", "associate", _entry()))
    assert checks["alias_routes"]["passed"] is False
    assert "no response" in checks["alias_routes"]["message"]


# --- PR #237 review findings ------------------------------------------------


def test_a_null_content_200_fails_instead_of_passing(monkeypatch) -> None:
    """Qodo #1: `content: null` is what a budget-exhausted thinking model returns.

    Measured live 2026-08-30 (`finish_reason: length`, `content: None`) — the
    exact condition this check exists to catch, and the one an earlier version
    reported healthy.
    """
    body = {"choices": [{"message": {"content": None}, "finish_reason": "length"}]}
    monkeypatch.setattr(rp, "_request", lambda *a, **k: (200, body, 0.4))
    check = rp.alias_check("http://gw", "associate", {}, 5.0)
    assert check["passed"] is False
    assert check["severity"] == "error"
    assert "EMPTY" in check["message"]


def test_a_non_string_content_fails_rather_than_crashing(monkeypatch) -> None:
    """Qodo #1: a list/dict content must not reach .strip() and raise."""
    for weird in ([{"type": "text", "text": "ok"}], {"text": "ok"}, 7):
        body = {"choices": [{"message": {"content": weird}}]}
        monkeypatch.setattr(rp, "_request", lambda *a, **k: (200, body, 0.1))
        check = rp.alias_check("http://gw", "associate", {}, 5.0)
        assert check["passed"] is False, f"{weird!r} should fail, not pass"


def test_a_malformed_200_body_fails(monkeypatch) -> None:
    for body in ({}, {"choices": []}, {"choices": [{}]}, None):
        monkeypatch.setattr(rp, "_request", lambda *a, **k: (200, body, 0.1))
        assert rp.alias_check("http://gw", "associate", {}, 5.0)["passed"] is False


def test_answer_text_only_accepts_a_real_string() -> None:
    assert rp._answer_text({"choices": [{"message": {"content": "ok"}}]}) == "ok"
    assert rp._answer_text({"choices": [{"message": {"content": None}}]}) is None
    assert rp._answer_text({"choices": [{"message": {"content": []}}]}) is None
    assert rp._answer_text({"choices": ["not-a-dict"]}) is None
    assert rp._answer_text(None) is None


def _serve(handler_body: bytes, status: int = 200, *, truncate: bool = False):
    """A real loopback HTTP server, so `_request`'s socket path is exercised."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class _H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - stdlib interface
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            # Promise more bytes than we send → the client raises IncompleteRead.
            self.send_header("Content-Length", str(len(handler_body) + (99 if truncate else 0)))
            self.end_headers()
            self.wfile.write(handler_body)

        def log_message(self, *a):  # silence the test log
            return

    srv = HTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


def test_request_returns_status_body_and_elapsed_over_a_real_socket() -> None:
    srv, base = _serve(b'{"max_model_len": 128000}')
    try:
        status, body, elapsed = rp._request(
            f"{base}/tokenize", payload={"model": "associate"}, headers={}, timeout=5.0
        )
    finally:
        srv.shutdown()
    assert status == 200
    assert body == {"max_model_len": 128000}
    assert elapsed >= 0


def test_request_reports_a_non_2xx_as_a_result_not_an_exception() -> None:
    srv, base = _serve(b'{"error": {"code": "model_not_found"}}', status=404)
    try:
        status, body, _ = rp._request(f"{base}/x", payload={}, headers={}, timeout=5.0)
    finally:
        srv.shutdown()
    assert status == 404
    assert body["error"]["code"] == "model_not_found"


def test_request_survives_a_truncated_body(monkeypatch) -> None:
    """Qodo #2: IncompleteRead must read as 'nothing answered', not abort."""
    srv, base = _serve(b'{"partial": true}', truncate=True)
    try:
        status, body, _ = rp._request(f"{base}/x", payload={}, headers={}, timeout=5.0)
    finally:
        srv.shutdown()
    assert status == 0
    assert body is None


def test_request_survives_an_undecodable_body() -> None:
    srv, base = _serve(b"not json at all")
    try:
        status, body, _ = rp._request(f"{base}/x", payload={}, headers={}, timeout=5.0)
    finally:
        srv.shutdown()
    assert status == 200
    assert body is None


def test_request_survives_a_dead_port() -> None:
    status, body, _ = rp._request("http://127.0.0.1:9/x", payload={}, headers={}, timeout=0.5)
    assert status == 0
    assert body is None


def test_a_tokenize_answer_without_a_window_is_a_warning(monkeypatch) -> None:
    """A 200 that omits max_model_len tells us nothing — say so, don't pass."""
    monkeypatch.setattr(rp, "_request", lambda *a, **k: (200, {"count": 2}, 0.01))
    check = rp.window_check("http://gw", "associate", 128000, {}, 5.0)
    assert check["passed"] is False
    assert "no max_model_len" in check["message"]


def test_a_routable_served_id_is_reported_as_such(monkeypatch) -> None:
    """When the raw id DOES route, say so — the ambiguity is deployment-specific."""
    monkeypatch.setattr(rp, "_request", lambda *a, **k: (200, _ok_completion(), 0.1))
    check = rp.served_id_check("http://gw", "associate", "nvidia/X", {}, 5.0)
    assert check["passed"] is True
    assert "also routes" in check["message"]
