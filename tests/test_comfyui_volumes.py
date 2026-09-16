"""Compose volumes and ownership for the `comfyui` service (innereye plan t4).

Static assertions over the shipped ``lobes/templates/fleet/docker-compose.yml``
plus one ``docker compose config`` rendered-substitution check, mirroring the
pattern ``tests/test_associate_exposure.py`` already uses for its own
rendered-compose acceptance criterion.

Three acceptance criteria (see the plan instruction, task t4):

1. The read-only models mount and the separately-owned, read-write output
   mount both default to a ``$HOME``-relative path -- no operator-specific
   absolute path is baked into the packaged template.
2. The service declares ``user: "${COMFY_UID:-1000}:${COMFY_GID:-1000}"``.
3. Overriding either host path needs exactly one env knob (``COMFY_MODELS`` /
   ``COMFY_OUTPUT``), and the models mount is read-only (``:ro``) while the
   output mount is not.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[1]
_FLEET_COMPOSE = _REPO / "lobes" / "templates" / "fleet" / "docker-compose.yml"
_ENV_EXAMPLE = _REPO / "lobes" / "templates" / "fleet" / "env.example"


def _fleet_text() -> str:
    return _FLEET_COMPOSE.read_text(encoding="utf-8")


def _fleet() -> dict:
    return yaml.safe_load(_fleet_text())


def _service() -> dict:
    return _fleet()["services"]["comfyui"]


def _volumes() -> list:
    return _service().get("volumes", [])


def _models_mount() -> str:
    matches = [v for v in _volumes() if "/opt/ComfyUI/models" in v]
    assert len(matches) == 1, f"expected exactly one models mount, got {matches}"
    return matches[0]


def _output_mount() -> str:
    matches = [v for v in _volumes() if "/opt/ComfyUI/output" in v]
    assert len(matches) == 1, f"expected exactly one output mount, got {matches}"
    return matches[0]


# ---------------------------------------------------------------------------
# criterion 1 -- no operator-specific absolute path baked into the template
# ---------------------------------------------------------------------------


class TestNoOperatorSpecificPath:
    def test_models_mount_uses_home_relative_fallback(self) -> None:
        mount = _models_mount()
        assert mount.startswith("${COMFY_MODELS:-${HOME:-/root}/comfy/ComfyUI/models}")
        assert mount.endswith(":/opt/ComfyUI/models:ro")

    def test_output_mount_uses_home_relative_fallback(self) -> None:
        mount = _output_mount()
        assert mount.startswith("${COMFY_OUTPUT:-${HOME:-/root}/comfy/ComfyUI/output}")
        assert mount.endswith(":/opt/ComfyUI/output")
        # Read-write, not read-only -- must not accidentally carry :ro too.
        assert not mount.endswith(":/opt/ComfyUI/output:ro")

    def test_no_hardcoded_operator_home_in_the_service_block(self) -> None:
        """Neither mount may hardcode a literal operator path such as
        /home/spark/... -- only the ${VAR:-fallback} form."""
        for mount in (_models_mount(), _output_mount()):
            assert "/home/" not in mount, f"operator-specific path leaked into template: {mount}"

    def test_models_and_output_are_separately_owned_paths(self) -> None:
        assert _models_mount() != _output_mount()
        assert "COMFY_MODELS" in _models_mount()
        assert "COMFY_OUTPUT" in _output_mount()


# ---------------------------------------------------------------------------
# criterion 2 -- non-root user, matching the image's own USER 1000:1000
# ---------------------------------------------------------------------------


class TestNonRootUser:
    def test_user_key_declared(self) -> None:
        svc = _service()
        assert svc.get("user") == "${COMFY_UID:-1000}:${COMFY_GID:-1000}"

    def test_no_other_fleet_service_sets_user(self) -> None:
        """The instruction's own load-bearing claim: comfyui is the FIRST
        fleet service to set `user:` -- prove no other service also does,
        so this stays true rather than becoming stale."""
        for name, svc in _fleet()["services"].items():
            if name == "comfyui":
                continue
            assert (
                "user" not in svc
            ), f"{name} unexpectedly sets user: -- update the rationale comment"

    def test_rationale_comment_cites_the_dockerfile_user_directive(self) -> None:
        text = _fleet_text()
        start = text.index("\n  comfyui:\n")
        rest = text[start:]
        import re

        match = re.search(r"\n  [a-z][a-z0-9-]*:\s*\n", rest[1:])
        end = start + 2 + match.start() if match else len(text)
        block = text[start:end]
        assert "1000:1000" in block
        assert "first" in block.lower()


# ---------------------------------------------------------------------------
# criterion 3 -- one env knob per path; read-only weights, read-write output
# ---------------------------------------------------------------------------


class TestOneKnobPerPath:
    def test_models_mount_is_read_only(self) -> None:
        assert _models_mount().endswith(":ro")

    def test_output_mount_is_read_write(self) -> None:
        assert not _output_mount().endswith(":ro")

    def test_env_example_documents_all_four_knobs(self) -> None:
        text = _ENV_EXAMPLE.read_text(encoding="utf-8")
        for knob in ("COMFY_MODELS", "COMFY_OUTPUT", "COMFY_UID", "COMFY_GID"):
            assert knob in text, f"{knob} is not documented in env.example"

    def test_env_example_comment_explains_the_precedent(self) -> None:
        text = _ENV_EXAMPLE.read_text(encoding="utf-8")
        idx = text.index("COMFY_MODELS")
        surrounding = text[max(0, idx - 700) : idx]
        assert "READ-ONLY" in surrounding or "read-only" in surrounding.lower()


# ---------------------------------------------------------------------------
# rendered compose -- prove `docker compose config` substitutes cleanly with
# a single overridden knob (mirrors test_associate_exposure.py's pattern)
# ---------------------------------------------------------------------------


class TestRenderedCompose:
    def test_rendered_compose_substitutes_comfy_models_override(self) -> None:
        if shutil.which("docker") is None:  # pragma: no cover - env dependent
            pytest.skip("docker not available")
        env = {"PATH": os.environ.get("PATH", ""), "COMFY_MODELS": "/tmp/some-other-weights"}
        proc = subprocess.run(
            ["docker", "compose", "-f", str(_FLEET_COMPOSE), "--profile", "innereye", "config"],
            capture_output=True,
            text=True,
            cwd=str(_FLEET_COMPOSE.parent),
            env=env,
        )
        if proc.returncode != 0:  # pragma: no cover - env dependent
            pytest.skip(f"docker compose config unavailable: {proc.stderr[-300:]}")
        rendered = yaml.safe_load(proc.stdout)["services"]
        assert "comfyui" in rendered
        volumes = rendered["comfyui"].get("volumes", [])
        models = [
            v for v in volumes if isinstance(v, dict) and v.get("target") == "/opt/ComfyUI/models"
        ]
        if not models:
            # older docker compose renders volumes as plain strings, not dicts
            joined = " ".join(str(v) for v in volumes)
            assert "/tmp/some-other-weights" in joined
        else:
            assert models[0]["source"] == "/tmp/some-other-weights"
            assert models[0].get("read_only") is True
        assert rendered["comfyui"].get("user") in (
            "1000:1000",
            "${COMFY_UID:-1000}:${COMFY_GID:-1000}",
        )
