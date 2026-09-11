"""``lobes doctor``'s ``pool_arming`` finding (#244 t6).

``lobes.gateway.server._check_pool_arming`` is a hard startup gate: a
deployment that drops a role (``<PREFIX>_FEASIBLE=false``) while declaring
the PLURAL peer channel (``<PREFIX>_PEER_ORIGINS``) but not the SINGULAR one
(``<PREFIX>_PEER_ORIGIN``) fails the gateway's own boot with
``ReplicaConfigError`` — silently losing ``hosted_by`` the moment the pool
arms. That guard is correct (server.py:3981-4028) and this task does not
relax it. Instead ``doctor`` reuses the SAME guard, offline, against the
deployed ``.env``, so the trap is caught before a real box flips into it —
not by a failed gateway boot.

This module proves:

* a Thor-shaped ``.env`` that drops ``primary`` (cortex) while declaring only
  ``PRIMARY_PEER_ORIGINS`` is caught by the new ``pool_arming`` check, naming
  both the plural and singular keys (mirroring
  ``_check_pool_arming``'s own error text);
* declaring the singular origin alongside the plural one clears the finding;
* a deployment that never declares any plural peer family passes trivially
  (no false positive on every pre-#199 deployment);
* the check is read-only — never invoked by ``--fix``/``--apply``.
"""

from __future__ import annotations

import json

from lobes.cli import main
from lobes.runtime import _compose, _env


def _scaffold_fleet(path, *, profile: str = "thor"):
    _compose.write_scaffold(path, force=True, templates=dict(_compose.FLEET_TEMPLATES))
    _compose.write_plugin_file(path, force=True)
    _env.set_env(path / ".env", "LOBES_PROFILE", profile)
    return path


def _doctor_json(capsys, *args: str) -> dict:
    main(["doctor", "--json", *args])
    return json.loads(capsys.readouterr().out)


def _checks(payload: dict) -> dict:
    return {c["id"]: c for c in payload["checks"]}


class TestPoolArmingTrap:
    def test_plural_without_singular_on_a_dropped_role_is_caught(
        self, tmp_path, monkeypatch, capsys
    ):
        _scaffold_fleet(tmp_path)
        monkeypatch.setenv("LOBES_DIR", str(tmp_path))
        monkeypatch.setattr(_compose, "docker_available", lambda: True)
        env_path = tmp_path / ".env"
        # The exact flip the task names: Thor drops cortex (primary) and
        # declares the plural replica family, but never the singular origin.
        _env.set_env(env_path, "PRIMARY_FEASIBLE", "false")
        _env.set_env(env_path, "PRIMARY_PEER_ORIGINS", "http://spark.example:8000")

        payload = _doctor_json(capsys)
        check = _checks(payload)["pool_arming"]
        assert check["passed"] is False
        assert check["severity"] == "error"
        assert "PRIMARY_PEER_ORIGINS" in check["message"]
        assert "PRIMARY_PEER_ORIGIN" in check["message"]

    def test_declaring_the_singular_origin_too_clears_the_finding(
        self, tmp_path, monkeypatch, capsys
    ):
        _scaffold_fleet(tmp_path)
        monkeypatch.setenv("LOBES_DIR", str(tmp_path))
        monkeypatch.setattr(_compose, "docker_available", lambda: True)
        env_path = tmp_path / ".env"
        _env.set_env(env_path, "PRIMARY_FEASIBLE", "false")
        _env.set_env(env_path, "PRIMARY_PEER_ORIGINS", "http://spark.example:8000")
        _env.set_env(env_path, "PRIMARY_PEER_ORIGIN", "http://spark.example:8000")

        payload = _doctor_json(capsys)
        check = _checks(payload)["pool_arming"]
        assert check["passed"] is True

    def test_no_plural_peer_family_declared_passes_trivially(self, tmp_path, monkeypatch, capsys):
        # Every pre-#199 deployment — no PEER_ORIGINS anywhere.
        _scaffold_fleet(tmp_path)
        monkeypatch.setenv("LOBES_DIR", str(tmp_path))
        monkeypatch.setattr(_compose, "docker_available", lambda: True)

        payload = _doctor_json(capsys)
        check = _checks(payload)["pool_arming"]
        assert check["passed"] is True

    def test_a_hosted_pool_needs_no_singular_origin(self, tmp_path, monkeypatch, capsys):
        # The #199 case: a box pools a role it HOSTS (feasible, not dropped)
        # — no referral to publish, so no singular origin is required.
        _scaffold_fleet(tmp_path)
        monkeypatch.setenv("LOBES_DIR", str(tmp_path))
        monkeypatch.setattr(_compose, "docker_available", lambda: True)
        env_path = tmp_path / ".env"
        _env.set_env(env_path, "PRIMARY_PEER_ORIGINS", "http://peer.example:8000")
        _env.set_env(env_path, "GATEWAY_SELF_ORIGIN", "http://this-box.example:8000")

        payload = _doctor_json(capsys)
        check = _checks(payload)["pool_arming"]
        assert check["passed"] is True

    def test_fix_apply_never_writes_a_pool_arming_remedy(self, tmp_path, monkeypatch, capsys):
        # --fix --apply's job is the pre-existing missing-only heal (absent
        # profile-required keys, etc.) — it may still append UNRELATED keys,
        # but it must never invent a PRIMARY_PEER_ORIGIN line to silence this
        # finding, and the finding must still fire afterwards: this trap is
        # not in --fix's remit, only doctor's read-only diagnosis.
        _scaffold_fleet(tmp_path)
        monkeypatch.setenv("LOBES_DIR", str(tmp_path))
        monkeypatch.setattr(_compose, "docker_available", lambda: True)
        env_path = tmp_path / ".env"
        _env.set_env(env_path, "PRIMARY_FEASIBLE", "false")
        _env.set_env(env_path, "PRIMARY_PEER_ORIGINS", "http://spark.example:8000")

        _doctor_json(capsys, "--fix", "--apply")

        after = _env.read_env_file(env_path)
        assert not (after.get("PRIMARY_PEER_ORIGIN") or "").strip()

        payload = _doctor_json(capsys)
        assert _checks(payload)["pool_arming"]["passed"] is False
