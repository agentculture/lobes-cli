"""``lobes doctor``'s mesh/peer-retirement findings (t9).

The mesh (``LOBES_MESH_KEY``/``NAME``/``SEEDS``) replaced the operator-typed
``<PREFIX>_PEER_*`` family. This module proves three new READ-ONLY findings:

* ``peer_family_retired`` — a leftover ``<PREFIX>_PEER_*`` key in ``.env``.
* ``mesh_key_shell_mismatch`` — the invoking shell's ``LOBES_MESH_KEY``
  disagrees with ``.env``'s.
* ``passthrough_missing`` — a set ``LOBES_MESH_*`` key with no gateway
  passthrough in ``docker-compose.yml``.

All three: absent on a clean deployment, never touched by ``--fix``.
"""

from __future__ import annotations

import json

from lobes.cli import main
from lobes.runtime import _compose, _env


def _scaffold_fleet(path, *, profile: str = "spark"):
    _compose.write_scaffold(path, force=True, templates=dict(_compose.FLEET_TEMPLATES))
    _compose.write_plugin_file(path, force=True)
    _env.set_env(path / ".env", "LOBES_PROFILE", profile)
    return path


def _doctor_json(capsys, *args: str) -> dict:
    main(["doctor", "--json", *args])
    return json.loads(capsys.readouterr().out)


class TestPeerFamilyRetired:
    def test_leftover_peer_origin_is_reported(self, tmp_path, monkeypatch, capsys):
        _scaffold_fleet(tmp_path)
        monkeypatch.setenv("LOBES_DIR", str(tmp_path))
        monkeypatch.setattr(_compose, "docker_available", lambda: True)
        _env.set_env(tmp_path / ".env", "PRIMARY_PEER_ORIGIN", "http://peer.example:8000")

        payload = _doctor_json(capsys)
        ids = {c["id"]: c for c in payload["checks"]}
        check = ids["peer_family_retired"]
        assert check["passed"] is False
        assert "PRIMARY_PEER_ORIGIN" in check["message"]
        assert "LOBES_MESH_KEY" in check["remediation"]

    def test_multiple_leftover_keys_all_listed_up_to_cap(self, tmp_path, monkeypatch, capsys):
        _scaffold_fleet(tmp_path)
        monkeypatch.setenv("LOBES_DIR", str(tmp_path))
        monkeypatch.setattr(_compose, "docker_available", lambda: True)
        _env.set_env(tmp_path / ".env", "PRIMARY_PEER_ORIGIN", "http://a.example:8000")
        _env.set_env(tmp_path / ".env", "WORKER_PEER_PROXY", "true")
        _env.set_env(tmp_path / ".env", "MUSE_PEER_API_KEY", "secret")

        payload = _doctor_json(capsys)
        ids = {c["id"]: c for c in payload["checks"]}
        check = ids["peer_family_retired"]
        assert check["passed"] is False
        assert "3 retired" in check["message"]

    def test_clean_env_reports_nothing(self, tmp_path, monkeypatch, capsys):
        _scaffold_fleet(tmp_path)
        monkeypatch.setenv("LOBES_DIR", str(tmp_path))
        monkeypatch.setattr(_compose, "docker_available", lambda: True)

        payload = _doctor_json(capsys)
        ids = {c["id"]: c for c in payload["checks"]}
        assert "peer_family_retired" not in ids

    def test_never_touched_by_fix_apply(self, tmp_path, monkeypatch, capsys):
        _scaffold_fleet(tmp_path)
        monkeypatch.setenv("LOBES_DIR", str(tmp_path))
        monkeypatch.setattr(_compose, "docker_available", lambda: True)
        _env.set_env(tmp_path / ".env", "PRIMARY_PEER_ORIGIN", "http://peer.example:8000")

        _doctor_json(capsys, "--fix", "--apply")

        # The key is never removed by --fix --apply (append-only heal, never
        # deletes an existing line) — the finding still fires afterward.
        payload = _doctor_json(capsys)
        ids = {c["id"]: c for c in payload["checks"]}
        assert ids["peer_family_retired"]["passed"] is False


class TestMeshKeyShellMismatch:
    def test_mismatch_reported(self, tmp_path, monkeypatch, capsys):
        _scaffold_fleet(tmp_path)
        monkeypatch.setenv("LOBES_DIR", str(tmp_path))
        monkeypatch.setattr(_compose, "docker_available", lambda: True)
        _env.set_env(tmp_path / ".env", "LOBES_MESH_KEY", "env-secret")
        monkeypatch.setenv("LOBES_MESH_KEY", "shell-secret")

        payload = _doctor_json(capsys)
        ids = {c["id"]: c for c in payload["checks"]}
        check = ids["mesh_key_shell_mismatch"]
        assert check["passed"] is False
        assert "LOBES_MESH_KEY" in check["message"]

    def test_matching_values_report_nothing(self, tmp_path, monkeypatch, capsys):
        _scaffold_fleet(tmp_path)
        monkeypatch.setenv("LOBES_DIR", str(tmp_path))
        monkeypatch.setattr(_compose, "docker_available", lambda: True)
        _env.set_env(tmp_path / ".env", "LOBES_MESH_KEY", "same-secret")
        monkeypatch.setenv("LOBES_MESH_KEY", "same-secret")

        payload = _doctor_json(capsys)
        ids = {c["id"]: c for c in payload["checks"]}
        assert "mesh_key_shell_mismatch" not in ids

    def test_shell_unset_reports_nothing(self, tmp_path, monkeypatch, capsys):
        _scaffold_fleet(tmp_path)
        monkeypatch.setenv("LOBES_DIR", str(tmp_path))
        monkeypatch.setattr(_compose, "docker_available", lambda: True)
        _env.set_env(tmp_path / ".env", "LOBES_MESH_KEY", "env-secret")
        monkeypatch.delenv("LOBES_MESH_KEY", raising=False)

        payload = _doctor_json(capsys)
        ids = {c["id"]: c for c in payload["checks"]}
        assert "mesh_key_shell_mismatch" not in ids


class TestPassthroughMissing:
    def test_missing_mesh_passthrough_is_reported(self, tmp_path, monkeypatch, capsys):
        _scaffold_fleet(tmp_path)
        monkeypatch.setenv("LOBES_DIR", str(tmp_path))
        monkeypatch.setattr(_compose, "docker_available", lambda: True)
        _env.set_env(tmp_path / ".env", "LOBES_MESH_KEY", "secret")
        compose_path = tmp_path / _compose.COMPOSE_FILE
        lines = compose_path.read_text(encoding="utf-8").splitlines()
        kept = [ln for ln in lines if "LOBES_MESH_KEY=${LOBES_MESH_KEY" not in ln]
        compose_path.write_text("\n".join(kept) + "\n", encoding="utf-8")

        payload = _doctor_json(capsys)
        ids = {c["id"]: c for c in payload["checks"]}
        check = ids["passthrough_missing"]
        assert check["passed"] is False
        assert "LOBES_MESH_KEY" in check["message"]

    def test_clean_env_reports_nothing(self, tmp_path, monkeypatch, capsys):
        _scaffold_fleet(tmp_path)
        monkeypatch.setenv("LOBES_DIR", str(tmp_path))
        monkeypatch.setattr(_compose, "docker_available", lambda: True)

        payload = _doctor_json(capsys)
        ids = {c["id"]: c for c in payload["checks"]}
        assert "passthrough_missing" not in ids

    def test_legacy_single_model_scaffold_has_no_finding(self, tmp_path, monkeypatch, capsys):
        _compose.write_scaffold(tmp_path, force=True)
        monkeypatch.setenv("LOBES_DIR", str(tmp_path))
        monkeypatch.setattr(_compose, "docker_available", lambda: True)
        _env.set_env(tmp_path / ".env", "LOBES_MESH_KEY", "secret")

        payload = _doctor_json(capsys)
        ids = {c["id"]: c for c in payload["checks"]}
        assert "passthrough_missing" not in ids

    def test_never_touched_by_fix_apply(self, tmp_path, monkeypatch, capsys):
        _scaffold_fleet(tmp_path)
        monkeypatch.setenv("LOBES_DIR", str(tmp_path))
        monkeypatch.setattr(_compose, "docker_available", lambda: True)
        _env.set_env(tmp_path / ".env", "LOBES_MESH_KEY", "secret")
        compose_path = tmp_path / _compose.COMPOSE_FILE
        lines = compose_path.read_text(encoding="utf-8").splitlines()
        kept = [ln for ln in lines if "LOBES_MESH_KEY=${LOBES_MESH_KEY" not in ln]
        compose_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        before = compose_path.read_bytes()

        _doctor_json(capsys, "--fix", "--apply")

        after = compose_path.read_bytes()
        assert after == before, "doctor --fix --apply must never edit docker-compose.yml"
