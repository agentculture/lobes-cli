"""Tests for ``lobes status`` fleet-awareness (issue #84).

Acceptance contract:
1. Fleet deployments: ``lobes status`` reports per-gear container states,
   never prints ``model-gear-vllm — not created``, and includes a pointer
   to ``lobes fleet status``.
2. Single-model deployments: output is byte-for-byte identical to the
   pre-fleet shape (no new keys, no new lines).
"""

from __future__ import annotations

import json
from pathlib import Path

from lobes.cli import main
from lobes.runtime import _compose, _env, _health

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GEAR_NAMES = (
    "model-gear-vllm-primary",
    "model-gear-vllm-multimodal",
    "model-gear-vllm-embed",
    "model-gear-vllm-rerank",
    "model-gear-gateway",
)


def _fake_deploy(tmp_path: Path, *, fleet: bool = False) -> Path:
    """Create a minimal deploy dir with docker-compose.yml and .env."""
    d = tmp_path / "deploy"
    d.mkdir()
    (d / _compose.COMPOSE_FILE).write_text("version: '3'\n")
    (d / _compose.ENV_FILE).write_text(
        "VLLM_MODEL=sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP\n"
        "VLLM_SERVED_NAME=qwen3.6-27b\n"
        "VLLM_PORT=8000\n"
        "VLLM_TOOL_CALL_PARSER=xml\n"
    )
    if fleet:
        (d / _compose.DOCKERFILE_GATEWAY).write_text("# gateway\n")
    return d


# ---------------------------------------------------------------------------
# Fleet tests
# ---------------------------------------------------------------------------


def test_status_fleet_json_shape(capsys, tmp_path, monkeypatch) -> None:
    """Fleet: --json output has deployment='fleet' and a containers list."""
    deploy = _fake_deploy(tmp_path, fleet=True)

    monkeypatch.setattr(_compose, "is_fleet", lambda d: True)
    monkeypatch.setattr(_compose, "fleet_containers", lambda d: _GEAR_NAMES)
    monkeypatch.setattr(
        _compose, "inspect_state", lambda name="model-gear-vllm": "running (healthy)"
    )
    monkeypatch.setattr(_health, "is_healthy", lambda port, timeout=3.0: True)
    monkeypatch.setattr(
        _env,
        "read_env",
        lambda path, key, default="(unset)": {
            "VLLM_MODEL": "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP",
            "VLLM_SERVED_NAME": "qwen3.6-27b",
            "VLLM_PORT": "8000",
            "VLLM_TOOL_CALL_PARSER": "xml",
        }.get(key, default),
    )

    rc = main(["status", "--json", "--compose-dir", str(deploy)])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert payload["deployment"] == "fleet"
    assert "containers" in payload
    assert isinstance(payload["containers"], list)
    names = [c["name"] for c in payload["containers"]]
    for gear in _GEAR_NAMES:
        assert gear in names, f"{gear!r} missing from containers list"
    # No "not created" anywhere in the JSON output
    assert "not created" not in out


def test_status_fleet_human_no_not_created(capsys, tmp_path, monkeypatch) -> None:
    """Fleet: human output must NOT contain 'not created' and must list gears."""
    deploy = _fake_deploy(tmp_path, fleet=True)

    monkeypatch.setattr(_compose, "is_fleet", lambda d: True)
    monkeypatch.setattr(_compose, "fleet_containers", lambda d: _GEAR_NAMES)
    monkeypatch.setattr(
        _compose, "inspect_state", lambda name="model-gear-vllm": "running (healthy)"
    )
    monkeypatch.setattr(_health, "is_healthy", lambda port, timeout=3.0: True)
    monkeypatch.setattr(
        _env,
        "read_env",
        lambda path, key, default="(unset)": {
            "VLLM_MODEL": "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP",
            "VLLM_SERVED_NAME": "qwen3.6-27b",
            "VLLM_PORT": "8000",
            "VLLM_TOOL_CALL_PARSER": "xml",
        }.get(key, default),
    )

    rc = main(["status", "--compose-dir", str(deploy)])
    assert rc == 0
    out = capsys.readouterr().out

    # Must NOT contain the old single-model container line
    # (the fleet gears share the "model-gear-vllm" prefix, so check the exact
    # old pattern: "state:  model-gear-vllm — ..." which is what the old code emitted)
    assert "state:  model-gear-vllm —" not in out
    assert "not created" not in out

    # Must list each fleet gear
    for gear in _GEAR_NAMES:
        assert gear in out, f"{gear!r} missing from human output"

    # Must have the pointer line
    assert "lobes fleet status" in out


# ---------------------------------------------------------------------------
# Single-model regression tests
# ---------------------------------------------------------------------------


def test_status_single_model_json_unchanged(capsys, tmp_path, monkeypatch) -> None:
    """Single-model: --json output must be byte-for-byte identical to the
    pre-fleet shape (same keys, no 'deployment' or 'containers' key)."""
    deploy = _fake_deploy(tmp_path, fleet=False)

    monkeypatch.setattr(_compose, "is_fleet", lambda d: False)
    monkeypatch.setattr(
        _compose, "inspect_state", lambda name="model-gear-vllm": "running (healthy)"
    )
    monkeypatch.setattr(_health, "is_healthy", lambda port, timeout=3.0: True)
    monkeypatch.setattr(
        _env,
        "read_env",
        lambda path, key, default="(unset)": {
            "VLLM_MODEL": "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP",
            "VLLM_SERVED_NAME": "qwen3.6-27b",
            "VLLM_PORT": "8000",
            "VLLM_TOOL_CALL_PARSER": "xml",
        }.get(key, default),
    )

    rc = main(["status", "--json", "--compose-dir", str(deploy)])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)

    # Exact keys the current code emits — no new keys (profile added for task t5)
    expected_keys = {
        "model",
        "served_name",
        "port",
        "tool_call_parser",
        "deployment_dir",
        "container",
        "state",
        "health",
        "profile",
    }
    assert (
        set(payload.keys()) == expected_keys
    ), f"Keys changed: got {set(payload.keys())}, expected {expected_keys}"
    # Must NOT have fleet keys
    assert "deployment" not in payload
    assert "containers" not in payload
    # Container must be the single-model name
    assert payload["container"] == "model-gear-vllm"


def test_status_single_model_human_unchanged(capsys, tmp_path, monkeypatch) -> None:
    """Single-model: human output must match the pre-fleet shape exactly."""
    deploy = _fake_deploy(tmp_path, fleet=False)

    monkeypatch.setattr(_compose, "is_fleet", lambda d: False)
    monkeypatch.setattr(
        _compose, "inspect_state", lambda name="model-gear-vllm": "running (healthy)"
    )
    monkeypatch.setattr(_health, "is_healthy", lambda port, timeout=3.0: True)
    monkeypatch.setattr(
        _env,
        "read_env",
        lambda path, key, default="(unset)": {
            "VLLM_MODEL": "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP",
            "VLLM_SERVED_NAME": "qwen3.6-27b",
            "VLLM_PORT": "8000",
            "VLLM_TOOL_CALL_PARSER": "xml",
        }.get(key, default),
    )

    rc = main(["status", "--compose-dir", str(deploy)])
    assert rc == 0
    out = capsys.readouterr().out

    # Must contain the single-model container line
    assert "model-gear-vllm" in out
    # Must NOT have fleet pointer
    assert "lobes fleet status" not in out
    # Must have the standard fields
    assert "model:" in out
    assert "served:" in out
    assert "parser:" in out
    assert "dir:" in out
    assert "state:" in out
    assert "health:" in out
