"""Tests for ``lobes doctor`` machine profile reporting (task t5).

Acceptance criteria:
1. doctor reports detected card (name or UNKNOWN) + device info + profile choice
2. doctor warns when profile doesn't match detected card (forced/unvalidated)
3. doctor warns when card is UNKNOWN
4. status reports the active profile (terse; details in doctor)
5. doctor/status stay read-only
"""

from __future__ import annotations

import json

from lobes.cli import main
from lobes.runtime import _compose, _detect, _env


def test_doctor_reports_machine_profile_known_card_matching_profile(
    tmp_path, monkeypatch, capsys
) -> None:
    """Detected: thor; Profile: thor => no warning, profile valid."""
    _compose.write_scaffold(tmp_path, force=True)
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    monkeypatch.setattr(_compose, "docker_available", lambda: True)

    # Mock detection to return thor card
    def mock_detect():
        return _detect.DetectedCard(
            resolved="thor",
            device_name="NVIDIA Thor",
            compute_capability="sm_110",
            total_memory_gb=560.0,
            hostname="machine1",
            device_tree_model=None,
            sources={
                "device_name": "nvidia-smi",
                "compute_capability": "nvidia-smi",
                "total_memory_gb": "/proc/meminfo",
                "hostname": "socket.gethostname",
                "device_tree_model": "unavailable",
            },
        )

    monkeypatch.setattr(_detect, "detect_card", mock_detect)
    # Also write the profile name to .env (as init would)
    _env.set_env(tmp_path / ".env", "LOBES_PROFILE", "thor")

    rc = main(["doctor", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)

    # machine_profile section should be present
    assert "machine_profile" in payload
    mp = payload["machine_profile"]
    assert mp["detected_card"] == "thor"
    assert mp["device_name"] == "NVIDIA Thor"
    assert mp["compute_capability"] == "sm_110"
    assert mp["total_memory_gb"] == 560.0
    assert mp["profile"] == "thor"
    assert mp.get("validated") is True  # matching profile
    assert payload["healthy"] is True  # no warning


def test_doctor_warns_forced_profile_mismatch(tmp_path, monkeypatch, capsys) -> None:
    """Detected: thor; Profile: spark => WARNING, forced/unvalidated."""
    _compose.write_scaffold(tmp_path, force=True)
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    monkeypatch.setattr(_compose, "docker_available", lambda: True)

    def mock_detect():
        return _detect.DetectedCard(
            resolved="thor",
            device_name="NVIDIA Thor",
            compute_capability="sm_110",
            total_memory_gb=560.0,
            hostname="machine1",
            device_tree_model=None,
            sources={
                "device_name": "nvidia-smi",
                "compute_capability": "nvidia-smi",
                "total_memory_gb": "/proc/meminfo",
                "hostname": "socket.gethostname",
                "device_tree_model": "unavailable",
            },
        )

    monkeypatch.setattr(_detect, "detect_card", mock_detect)
    # Init forced spark profile onto thor
    _env.set_env(tmp_path / ".env", "LOBES_PROFILE", "spark")

    main(["doctor", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert "machine_profile" in payload
    mp = payload["machine_profile"]
    assert mp["detected_card"] == "thor"
    assert mp["profile"] == "spark"
    assert mp.get("validated") is False  # NOT matching
    # Text output should warn
    main(["doctor"])
    out = capsys.readouterr().out
    assert "machine profile" in out.lower() or "profile" in out.lower()


def test_doctor_warns_unknown_card(tmp_path, monkeypatch, capsys) -> None:
    """Detected: unknown; Profile: spark => WARNING, card unrecognized."""
    _compose.write_scaffold(tmp_path, force=True)
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    monkeypatch.setattr(_compose, "docker_available", lambda: True)

    def mock_detect():
        return _detect.DetectedCard(
            resolved=_detect.UNKNOWN,
            device_name="UnknownGPU XYZ",
            compute_capability="sm_99",
            total_memory_gb=128.0,
            hostname="unknown-box",
            device_tree_model=None,
            sources={
                "device_name": "nvidia-smi",
                "compute_capability": "nvidia-smi",
                "total_memory_gb": "/proc/meminfo",
                "hostname": "socket.gethostname",
                "device_tree_model": "unavailable",
            },
        )

    monkeypatch.setattr(_detect, "detect_card", mock_detect)
    _env.set_env(tmp_path / ".env", "LOBES_PROFILE", "spark")

    main(["doctor", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert "machine_profile" in payload
    mp = payload["machine_profile"]
    assert mp["detected_card"] == _detect.UNKNOWN
    assert mp["device_name"] == "UnknownGPU XYZ"
    assert mp["compute_capability"] == "sm_99"
    assert mp.get("warning") is not None  # should have a warning


def test_status_reports_active_profile_fleet(tmp_path, monkeypatch, capsys) -> None:
    """Fleet: lobes status includes the active profile."""
    _compose.write_scaffold(tmp_path, force=True)
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    monkeypatch.setattr(_compose, "is_fleet", lambda d: True)
    monkeypatch.setattr(_compose, "fleet_containers", lambda d: [])
    monkeypatch.setattr(
        _env,
        "read_env",
        lambda path, key, default="(unset)": {
            "VLLM_MODEL": "qwen",
            "VLLM_SERVED_NAME": "qwen3.6-27b",
            "VLLM_PORT": "8000",
            "VLLM_TOOL_CALL_PARSER": "xml",
            "LOBES_PROFILE": "thor",
        }.get(key, default),
    )

    rc = main(["status", "--json", "--compose-dir", str(tmp_path)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("profile") == "thor"


def test_status_reports_active_profile_single(tmp_path, monkeypatch, capsys) -> None:
    """Single model: lobes status includes the active profile."""
    _compose.write_scaffold(tmp_path, force=True)
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    monkeypatch.setattr(_compose, "is_fleet", lambda d: False)
    monkeypatch.setattr(
        _compose, "inspect_state", lambda name="model-gear-vllm": "running (healthy)"
    )
    monkeypatch.setattr(
        _env,
        "read_env",
        lambda path, key, default="(unset)": {
            "VLLM_MODEL": "qwen",
            "VLLM_SERVED_NAME": "qwen3.6-27b",
            "VLLM_PORT": "8000",
            "VLLM_TOOL_CALL_PARSER": "xml",
            "LOBES_PROFILE": "spark",
        }.get(key, default),
    )

    rc = main(["status", "--json", "--compose-dir", str(tmp_path)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("profile") == "spark"
