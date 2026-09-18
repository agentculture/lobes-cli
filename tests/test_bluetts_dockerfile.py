"""Static assertions for lobes/templates/fleet/Dockerfile.bluetts
(hebrew-realtime plan, approved deviation d8). Pure file-content checks — no
docker daemon, no network — the same contract as the sibling Dockerfile tests.
"""

from __future__ import annotations

import re
from pathlib import Path

_TEMPLATES = Path(__file__).resolve().parents[1] / "lobes" / "templates" / "fleet"
_DOCKERFILE = _TEMPLATES / "Dockerfile.bluetts"


def _instructions() -> str:
    text = _DOCKERFILE.read_text(encoding="utf-8")
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))


def test_entrypoint_is_the_packaged_sidecar_module() -> None:
    assert "lobes.realtime.bluetts_server" in _instructions()


def test_engine_code_is_pinned_to_a_full_commit_sha() -> None:
    assert re.search(r"ARG BLUETTS_REF=[0-9a-f]{40}\b", _instructions())


def test_no_weights_are_fetched_or_named_at_build_time() -> None:
    """The weights repo declares no licence (2026-09-18): the image must not
    bake or download them — the operator mounts a local directory."""
    body = _instructions().lower()
    for needle in (
        "hf download",
        "huggingface-cli",
        "snapshot_download",
        "notmax123",
        "onnx_models",
    ):
        assert needle not in body, needle


def test_cpu_only_image() -> None:
    body = _instructions().lower()
    assert "nvidia/cuda" not in body
    assert "onnxruntime-gpu" not in body


def test_uv_is_the_installer() -> None:
    body = _instructions()
    assert "uv pip install --system" in body
    assert not re.search(r"(?<!uv )\bpip install\b", body)


def test_lobes_is_installed_with_the_bluetts_extra_at_the_pinned_version() -> None:
    assert '"lobes-cli[bluetts]==${MODEL_GEAR_VERSION}"' in _instructions()


def test_same_internal_port_as_the_service_it_replaces() -> None:
    assert "EXPOSE 9000" in _instructions()


def test_dockerfile_ships_in_the_wheel() -> None:
    assert _DOCKERFILE.exists()


def test_engine_dependencies_come_from_the_engines_own_lockfile() -> None:
    """An unpinned resolve broke the G2P at warm-up (renikud-plus 0.5.0)."""
    body = _instructions()
    assert "uv export --frozen" in body
    assert "--no-deps /opt/bluetts" in body
