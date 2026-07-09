"""Tests for ``lobes doctor`` — real docker/compose/.env/health checks."""

from __future__ import annotations

import json

from lobes import __version__
from lobes.cli import main
from lobes.runtime import _compose, _env, _health


def test_doctor_offline_is_unhealthy(capsys) -> None:
    # Offline fixture: docker unavailable + no deployment scaffolded.
    rc = main(["doctor", "--json"])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    ids = {c["id"]: c for c in payload["checks"]}
    assert ids["docker_available"]["passed"] is False
    assert ids["compose_present"]["passed"] is False
    assert payload["healthy"] is False


def test_doctor_healthy_when_docker_and_scaffold_present(tmp_path, monkeypatch, capsys) -> None:
    _compose.write_scaffold(tmp_path, force=True)
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    monkeypatch.setattr(_compose, "docker_available", lambda: True)

    rc = main(["doctor", "--json"])
    payload = json.loads(capsys.readouterr().out)
    ids = {c["id"]: c for c in payload["checks"]}
    assert ids["docker_available"]["passed"] is True
    assert ids["compose_present"]["passed"] is True
    assert "env_coherence" in ids
    # The scaffolded VLLM_SERVED_NAME matches culture.yaml in this repo.
    assert ids["env_coherence"]["passed"] is True
    # /health is down (info severity only) → run is still healthy overall.
    assert payload["healthy"] is True
    assert rc == 0


def test_doctor_compose_dir_flag(tmp_path, monkeypatch, capsys) -> None:
    # Default home is the empty tmp dir (autouse); --compose-dir points elsewhere.
    _compose.write_scaffold(tmp_path, force=True)
    monkeypatch.setattr(_compose, "docker_available", lambda: True)
    rc = main(["doctor", "--compose-dir", str(tmp_path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    ids = {c["id"]: c for c in payload["checks"]}
    assert ids["compose_present"]["passed"] is True
    assert "env_coherence" in ids  # only reachable once a dir resolves
    assert rc == 0


def test_doctor_env_mismatch_warns_but_passes(tmp_path, monkeypatch, capsys) -> None:
    _compose.write_scaffold(tmp_path, force=True)
    _env.set_env(tmp_path / ".env", "VLLM_SERVED_NAME", "other/model")
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    monkeypatch.setattr(_compose, "docker_available", lambda: True)

    rc = main(["doctor", "--json"])
    payload = json.loads(capsys.readouterr().out)
    ids = {c["id"]: c for c in payload["checks"]}
    assert ids["env_coherence"]["passed"] is False
    assert ids["env_coherence"]["severity"] == "warn"
    # A warn does not fail the run.
    assert payload["healthy"] is True
    assert rc == 0


# ---------------------------------------------------------------------------
# gateway_version_match — issue #99 (the structural cause of issue #92):
# Dockerfile.gateway pins MODEL_GEAR_VERSION once at `lobes init` time and
# nothing ever re-pins it, so a deployed gateway container can silently run a
# stale lobes-cli release long after the host CLI (and PyPI) moved on.
# ---------------------------------------------------------------------------


def test_doctor_version_check_degrades_when_gateway_unreachable(
    tmp_path, monkeypatch, capsys
) -> None:
    """Unreachable must NOT be a false pass, and must NOT fail the run either
    — a down gateway is ordinary here (the health check treats it the same
    way), not evidence of a real skew defect."""
    _compose.write_scaffold(tmp_path, force=True)
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    monkeypatch.setattr(_compose, "docker_available", lambda: True)
    # conftest's autouse fixture already stubs _health.fetch_health to None
    # (unreachable) by default — this test relies on that default explicitly.

    rc = main(["doctor", "--json"])
    payload = json.loads(capsys.readouterr().out)
    ids = {c["id"]: c for c in payload["checks"]}
    check = ids["gateway_version_match"]
    assert check["passed"] is False
    assert check["severity"] == "info"  # non-fatal — never contributes to healthy=False
    assert "cannot verify" in check["message"]
    assert payload["healthy"] is True  # unreachable alone must not fail the run
    assert rc == 0


def test_doctor_version_check_degrades_when_gateway_reports_no_version(
    tmp_path, monkeypatch, capsys
) -> None:
    """A reachable gateway from before issue #99 (no `version` in /health) is
    also "cannot verify", not a mismatch and not a match — same non-fatal
    info severity as fully unreachable."""
    _compose.write_scaffold(tmp_path, force=True)
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    monkeypatch.setattr(_compose, "docker_available", lambda: True)
    monkeypatch.setattr(_health, "fetch_health", lambda *a, **k: {"status": "ok"})

    rc = main(["doctor", "--json"])
    payload = json.loads(capsys.readouterr().out)
    ids = {c["id"]: c for c in payload["checks"]}
    check = ids["gateway_version_match"]
    assert check["passed"] is False
    assert check["severity"] == "info"
    assert payload["healthy"] is True
    assert rc == 0


def test_doctor_version_match_passes_at_error_severity(tmp_path, monkeypatch, capsys) -> None:
    _compose.write_scaffold(tmp_path, force=True)
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    monkeypatch.setattr(_compose, "docker_available", lambda: True)
    monkeypatch.setattr(
        _health, "fetch_health", lambda *a, **k: {"status": "ok", "version": __version__}
    )

    rc = main(["doctor", "--json"])
    payload = json.loads(capsys.readouterr().out)
    ids = {c["id"]: c for c in payload["checks"]}
    check = ids["gateway_version_match"]
    assert check["passed"] is True
    assert check["severity"] == "error"  # class of failure this check guards, not a claim of one
    assert __version__ in check["message"]
    assert payload["healthy"] is True
    assert rc == 0


def test_doctor_version_mismatch_names_both_versions_and_fails_the_run(
    tmp_path, monkeypatch, capsys
) -> None:
    """Issue #99's core assertion: a version mismatch is a real, actionable
    defect — passed=False, severity="error" (fails the overall run), the
    message names BOTH versions, and the remediation names the exact fix."""
    _compose.write_scaffold(tmp_path, force=True)
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    monkeypatch.setattr(_compose, "docker_available", lambda: True)
    monkeypatch.setattr(
        _health, "fetch_health", lambda *a, **k: {"status": "ok", "version": "0.1.0"}
    )

    rc = main(["doctor", "--json"])
    payload = json.loads(capsys.readouterr().out)
    ids = {c["id"]: c for c in payload["checks"]}
    check = ids["gateway_version_match"]
    assert check["passed"] is False
    assert check["severity"] == "error"
    assert "0.1.0" in check["message"]
    assert __version__ in check["message"]
    assert f"MODEL_GEAR_VERSION={__version__}" in check["remediation"]
    assert f"{tmp_path}/.env" in check["remediation"] or str(tmp_path) in check["remediation"]
    assert "docker compose up -d --build gateway" in check["remediation"]
    # A real skew defect DOES fail the overall run — that is the whole point.
    assert payload["healthy"] is False
    assert rc == 1


def test_doctor_version_mismatch_remediation_names_placeholder_when_unscaffolded(
    monkeypatch,
) -> None:
    """No deployment resolved → the remediation still names the exact fix,
    with a generic <deployment> placeholder instead of a concrete path."""
    from lobes.cli._commands import doctor as doctor_module

    monkeypatch.setattr(
        _health, "fetch_health", lambda *a, **k: {"status": "ok", "version": "0.1.0"}
    )
    report = doctor_module._diagnose(None)
    ids = {c["id"]: c for c in report["checks"]}
    check = ids["gateway_version_match"]
    assert check["passed"] is False
    assert "<deployment>/.env" in check["remediation"]
