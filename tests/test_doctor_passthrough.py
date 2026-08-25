"""``lobes doctor``'s "gateway passthrough" finding (issue #199, t3).

A ``.env`` key with no matching passthrough line in the deployed
``docker-compose.yml``'s gateway service silently never reaches the gateway
container — this bit the mesh on 2026-07-17 (the ``MUSE_*`` keys were set in
``.env`` but the compose file carried no passthrough for them, and every
other check, including ``/health``, stayed green). This module proves the
new ``gateway_passthrough`` doctor check catches that class of defect, that
the packaged template itself never trips it, and that ``--fix --apply``
never edits ``docker-compose.yml`` to "fix" it (a compose gap is healed by
re-scaffolding, not patching).
"""

from __future__ import annotations

import json

from lobes.cli import main
from lobes.cli._commands import doctor as doctor_module
from lobes.runtime import _compose, _env


def _scaffold_fleet(path, *, profile: str = "spark"):
    """A complete fleet deployment, as ``lobes init --apply`` leaves it."""
    _compose.write_scaffold(path, force=True, templates=dict(_compose.FLEET_TEMPLATES))
    _compose.write_plugin_file(path, force=True)
    _env.set_env(path / ".env", "LOBES_PROFILE", profile)
    return path


def _doctor_json(capsys, *args: str) -> dict:
    main(["doctor", "--json", *args])
    return json.loads(capsys.readouterr().out)


def _drop_compose_line(compose_path, key: str) -> None:
    """Remove the ``- KEY=${KEY...}`` line for ``key`` from the gateway block
    — simulating a deployed compose file that predates that key's passthrough
    (exactly the 2026-07-17 MUSE_* shape)."""
    lines = compose_path.read_text(encoding="utf-8").splitlines()
    kept = [ln for ln in lines if f"{key}=${{{key}" not in ln]
    compose_path.write_text("\n".join(kept) + "\n", encoding="utf-8")


class TestMissingPassthroughIsFlagged:
    def test_missing_primary_peer_origins_is_reported(self, tmp_path, monkeypatch, capsys):
        _scaffold_fleet(tmp_path)
        monkeypatch.setenv("LOBES_DIR", str(tmp_path))
        monkeypatch.setattr(_compose, "docker_available", lambda: True)
        _env.set_env(tmp_path / ".env", "PRIMARY_PEER_ORIGINS", "http://peer-a.example:8000")
        _drop_compose_line(tmp_path / _compose.COMPOSE_FILE, "PRIMARY_PEER_ORIGINS")

        payload = _doctor_json(capsys)
        ids = {c["id"]: c for c in payload["checks"]}
        check = ids["gateway_passthrough"]
        assert check["passed"] is False
        assert "PRIMARY_PEER_ORIGINS" in check["message"]
        # A compose gap is healed by re-scaffolding, never doctor --fix.
        assert "init --apply" in check["remediation"]

    def test_doctor_fix_apply_leaves_compose_byte_identical(self, tmp_path, monkeypatch, capsys):
        _scaffold_fleet(tmp_path)
        monkeypatch.setenv("LOBES_DIR", str(tmp_path))
        monkeypatch.setattr(_compose, "docker_available", lambda: True)
        _env.set_env(tmp_path / ".env", "PRIMARY_PEER_ORIGINS", "http://peer-a.example:8000")
        compose_path = tmp_path / _compose.COMPOSE_FILE
        _drop_compose_line(compose_path, "PRIMARY_PEER_ORIGINS")
        before = compose_path.read_bytes()

        _doctor_json(capsys, "--fix", "--apply")

        after = compose_path.read_bytes()
        assert after == before, "doctor --fix --apply must never edit docker-compose.yml"

        # The finding still fires post-heal — the heal was files/env-only and
        # never touched the compose gap.
        payload = _doctor_json(capsys)
        ids = {c["id"]: c for c in payload["checks"]}
        assert ids["gateway_passthrough"]["passed"] is False


class TestUnsetKeysNeverFlagged:
    def test_unset_gateway_keys_pass_clean(self, tmp_path, monkeypatch, capsys):
        """A freshly scaffolded fleet deployment sets none of the new
        replica-pool/fingerprint knobs — the check must not fire on keys the
        operator never set."""
        _scaffold_fleet(tmp_path)
        monkeypatch.setenv("LOBES_DIR", str(tmp_path))
        monkeypatch.setattr(_compose, "docker_available", lambda: True)

        payload = _doctor_json(capsys)
        ids = {c["id"]: c for c in payload["checks"]}
        check = ids["gateway_passthrough"]
        assert check["passed"] is True
        assert payload["healthy"] is True


class TestSingleModelDeploymentSkipsTheCheck:
    def test_legacy_single_model_scaffold_has_no_finding(self, tmp_path, monkeypatch, capsys):
        _compose.write_scaffold(tmp_path, force=True)  # legacy single-model set
        monkeypatch.setenv("LOBES_DIR", str(tmp_path))
        monkeypatch.setattr(_compose, "docker_available", lambda: True)

        payload = _doctor_json(capsys)
        ids = {c["id"]: c for c in payload["checks"]}
        # The legacy scaffold has no gateway service and no profile render at
        # all — like scaffold_files/profile_staleness, the check is omitted
        # entirely rather than firing on a service that was never expected to
        # exist.
        assert "gateway_passthrough" not in ids


class TestKeyEnumerationCoversAllNinePrefixes:
    def test_all_nine_role_prefixes_present(self):
        prefixes = set(doctor_module._GATEWAY_ROLE_PREFIXES)
        assert prefixes == {
            "PRIMARY",
            "MULTIMODAL",
            "MUSE",
            "WORKER",
            "HAND",
            "EMBED",
            "RERANK",
            "STT",
            "TTS",
        }

    def test_plural_peer_family_and_self_origin_are_relevant_keys(self):
        keys = set(doctor_module._gateway_relevant_keys())
        assert "GATEWAY_SELF_ORIGIN" in keys
        for prefix in doctor_module._GATEWAY_ROLE_PREFIXES:
            assert f"{prefix}_PEER_ORIGINS" in keys
            assert f"{prefix}_PEER_API_KEYS" in keys
            assert f"{prefix}_QUANTIZATION" in keys
            assert f"{prefix}_KV_CACHE_DTYPE" in keys
            assert f"{prefix}_REASONING_PARSER" in keys
            assert f"{prefix}_TOOL_CALL_PARSER" in keys
            assert f"{prefix}_SPECULATIVE_CONFIG" in keys


class TestFunctionToleratesNonFleetAndAbsentCompose:
    """Direct unit coverage of :func:`_gateway_passthrough_check`'s two
    degrade-to-pass branches — both are unreachable through the CLI today
    (``_diagnose`` only calls it inside the existing fleet-only block), but
    the function is written to be safe standalone too."""

    def test_non_fleet_dir_degrades_to_pass(self, tmp_path):
        _compose.write_scaffold(tmp_path, force=True)  # legacy, no Dockerfile.gateway
        check = doctor_module._gateway_passthrough_check(tmp_path)
        assert check["passed"] is True

    def test_absent_compose_file_degrades_to_pass(self, tmp_path):
        _scaffold_fleet(tmp_path)
        (tmp_path / _compose.COMPOSE_FILE).unlink()
        check = doctor_module._gateway_passthrough_check(tmp_path)
        assert check["passed"] is True


class TestPackagedTemplatePassesEveryKeyThrough:
    """The packaged template itself must never trip this finding — every key
    the check enumerates has a ``${VAR:-}``-shaped passthrough line in the
    gateway service already."""

    def test_no_gap_in_the_shipped_compose_template(self, tmp_path, monkeypatch, capsys):
        _scaffold_fleet(tmp_path)
        # Set every relevant key so the check actually exercises every one of
        # them, rather than trivially passing because nothing is set.
        for key in doctor_module._gateway_relevant_keys():
            _env.set_env(tmp_path / ".env", key, "x")
        monkeypatch.setenv("LOBES_DIR", str(tmp_path))
        monkeypatch.setattr(_compose, "docker_available", lambda: True)

        payload = _doctor_json(capsys)
        ids = {c["id"]: c for c in payload["checks"]}
        check = ids["gateway_passthrough"]
        assert check["passed"] is True, check["message"]


def test_passthrough_in_override_overlay_counts(tmp_path, monkeypatch):
    """An operator override.yml carrying the passthrough satisfies the check."""
    from lobes.cli._commands import doctor as D

    (tmp_path / "docker-compose.yml").write_text(
        "services:\n  gateway:\n    environment:\n      - GATEWAY_API_KEY=${GATEWAY_API_KEY:-}\n",
        encoding="utf-8",
    )
    (tmp_path / "docker-compose.override.yml").write_text(
        "services:\n  gateway:\n    environment:\n"
        "      - PRIMARY_PEER_ORIGINS=${PRIMARY_PEER_ORIGINS:-}\n",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text(
        "PRIMARY_PEER_ORIGINS=http://peer-a.example:8000\n", encoding="utf-8"
    )
    monkeypatch.setattr(D._compose, "is_fleet", lambda _d: True)
    result = D._gateway_passthrough_check(tmp_path)
    assert result["passed"] is True, result


def test_passthrough_under_another_service_does_not_count(tmp_path, monkeypatch):
    """Only services.gateway.environment propagates a key to the gateway (Qodo, PR #213)."""
    from lobes.cli._commands import doctor as D

    (tmp_path / "docker-compose.yml").write_text(
        "services:\n  gateway:\n    environment:\n      - GATEWAY_API_KEY=${GATEWAY_API_KEY:-}\n"
        "  vllm-primary:\n    environment:\n      - PRIMARY_PEER_ORIGINS=${PRIMARY_PEER_ORIGINS:-}\n",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text(
        "PRIMARY_PEER_ORIGINS=http://peer-a.example:8000\n", encoding="utf-8"
    )
    monkeypatch.setattr(D._compose, "is_fleet", lambda _d: True)
    result = D._gateway_passthrough_check(tmp_path)
    assert result["passed"] is False, result
    assert "PRIMARY_PEER_ORIGINS" in result["message"]


def test_gateway_environment_block_scan():
    from lobes.cli._commands.doctor import _gateway_environment_block

    text = (
        "services:\n"
        "  gateway:\n"
        "    image: x\n"
        "    environment:\n"
        "      # comment\n"
        "      - A=${A:-}\n"
        "      - B=${B}\n"
        "    ports:\n"
        "      - '8001:8000'\n"
        "  other:\n"
        "    environment:\n"
        "      - C=${C:-}\n"
    )
    assert _gateway_environment_block(text) == "- A=${A:-}\n- B=${B}"
