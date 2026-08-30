"""`lobes doctor --role` at the COMMAND level (issue #234, PR #237 review).

The probe's own decisions live in `test_doctor_role_probe.py`; this covers the
verb that wires them — flag validation, the 401 path, exit codes and rendering.
"""

from __future__ import annotations

import argparse

import pytest

import lobes.cli._commands.doctor as doc
from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError


def _args(**kw) -> argparse.Namespace:
    base = {"role": None, "fix": False, "apply": False, "json": False, "compose_dir": None}
    base.update(kw)
    return argparse.Namespace(**base)


def _stub_probe(monkeypatch, checks, *, registry=None):
    monkeypatch.setattr(doc._runtime_ops, "resolve_port_soft", lambda a: (8000, None))
    monkeypatch.setattr(doc._runtime_ops, "gateway_auth_headers", lambda d: {})
    monkeypatch.setattr(doc, "_fetch_gateway_capabilities", lambda p, headers=None: registry)
    monkeypatch.setattr(doc._role_probe, "probe_role", lambda *a, **k: checks)


def test_apply_without_fix_is_refused_even_with_a_role() -> None:
    """Qodo #3: the role branch must not swallow an invalid flag combination."""
    with pytest.raises(ModelGearError) as exc:
        doc.cmd_doctor(_args(role="associate", apply=True))
    assert exc.value.code == EXIT_USER_ERROR
    assert "--apply requires --fix" in exc.value.message


def test_role_with_fix_is_refused_rather_than_silently_ignored() -> None:
    """Qodo #3: `--role` probes a running lane; `--fix` heals files. Not both."""
    with pytest.raises(ModelGearError) as exc:
        doc.cmd_doctor(_args(role="associate", fix=True))
    assert "cannot be combined" in exc.value.message
    assert "associate" in exc.value.remediation


def test_an_unknown_role_is_a_user_error_naming_the_valid_ones() -> None:
    with pytest.raises(ModelGearError) as exc:
        doc.cmd_doctor(_args(role="nonsense"))
    assert exc.value.code == EXIT_USER_ERROR
    assert "cortex" in exc.value.remediation


def test_a_healthy_probe_exits_zero(monkeypatch, capsys) -> None:
    _stub_probe(monkeypatch, [doc._check("alias_routes", True, "info", "answered 200")])
    assert doc.cmd_doctor(_args(role="associate")) == 0
    assert "healthy" in capsys.readouterr().out


def test_an_error_check_exits_one(monkeypatch, capsys) -> None:
    _stub_probe(monkeypatch, [doc._check("alias_routes", False, "error", "refused")])
    assert doc.cmd_doctor(_args(role="associate")) == 1
    assert "unhealthy" in capsys.readouterr().out


def test_a_warn_check_does_not_fail_the_verb(monkeypatch) -> None:
    """A served id that does not route is guidance, not a broken deployment."""
    _stub_probe(monkeypatch, [doc._check("served_id", True, "warn", "use the alias")])
    assert doc.cmd_doctor(_args(role="associate")) == 0


def test_json_mode_emits_the_structured_report(monkeypatch, capsys) -> None:
    import json

    _stub_probe(monkeypatch, [doc._check("alias_routes", True, "info", "ok")])
    doc.cmd_doctor(_args(role="associate", json=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload["role"] == "associate"
    assert payload["healthy"] is True
    assert payload["checks"][0]["id"] == "alias_routes"
    assert payload["endpoint"].startswith("http://")


def test_a_401_reaches_the_friendly_wrapper(monkeypatch) -> None:
    """Qodo #4: an auth failure must give the .env remediation, not a traceback."""
    import urllib.error

    def _boom(port, headers=None):
        raise urllib.error.HTTPError("http://x", 401, "Unauthorized", {}, None)

    monkeypatch.setattr(doc._runtime_ops, "resolve_port_soft", lambda a: (8000, None))
    monkeypatch.setattr(doc._runtime_ops, "gateway_auth_headers", lambda d: {})
    monkeypatch.setattr(doc, "_fetch_gateway_capabilities", _boom)
    with pytest.raises(ModelGearError) as exc:
        doc.cmd_doctor(_args(role="associate"))
    # The wrapper's actionable message, not a raw HTTPError.
    assert "401" in str(exc.value.message) or "key" in str(exc.value.message).lower()


def test_an_unreachable_gateway_still_probes(monkeypatch) -> None:
    """A None registry is 'no advert', not a crash — the probe says so itself."""
    _stub_probe(
        monkeypatch,
        [doc._check("role_advert", False, "error", "no entry")],
        registry=None,
    )
    assert doc.cmd_doctor(_args(role="associate")) == 1
