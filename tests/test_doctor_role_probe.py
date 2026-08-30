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
    assert check["passed"] is False and check["severity"] == "error"
    assert "1048576" in check["message"] and "128000" in check["message"]


def test_a_matching_window_passes(monkeypatch) -> None:
    monkeypatch.setattr(rp, "_request", lambda *a, **k: (200, {"max_model_len": 128000}, 0.01))
    assert rp.window_check("http://gw", "associate", 128000, {}, 5.0)["passed"] is True


def test_an_empty_200_is_a_failure_not_a_healthy_lane(monkeypatch) -> None:
    """The measured caller hazard: HTTP 200 carrying no content looks like success."""
    monkeypatch.setattr(rp, "_request", lambda *a, **k: (200, _ok_completion(""), 0.4))
    check = rp.alias_check("http://gw", "associate", {}, 5.0)
    assert check["passed"] is False and check["severity"] == "error"
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
    assert check["passed"] is True and "0.19s" in check["message"]


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
