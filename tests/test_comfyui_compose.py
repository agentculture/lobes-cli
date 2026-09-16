"""The `comfyui` compose service (innereye plan t3, issue #82).

Static assertions over the shipped ``lobes/templates/fleet/docker-compose.yml``
— no docker daemon, no network, no image build (the same contract as
``tests/test_comfyui_dockerfile.py`` and ``tests/test_associate_exposure.py``).

Three acceptance criteria, one per test class:

1. **Expose-only isolation.** The service binds ``--listen 0.0.0.0 --port
   8188`` in-container and declares ``expose: ["8188"]`` with NO ``ports:``
   key. A rationale comment carries innereye's loopback finding (c36) forward
   in meaning: ComfyUI ships no authn, the bare venv answered that with
   ``--listen 127.0.0.1``, and a container answers the SAME property by
   publishing no host port instead (copying the loopback bind literally would
   make the lane unreachable from the gateway).
2. **A bespoke healthcheck.** ComfyUI has no ``/health`` route, so the
   Docker-level healthcheck probes ``/object_info`` (or ``/``) instead, and
   reports unhealthy during ComfyUI's own startup window rather than healthy
   immediately. The shared ``/health`` literal in ``lobes/runtime/_health.py``
   (the CLI-side probe) and the gateway's now-per-backend
   ``gateway/_readiness.py`` are both untouched by this task.
3. **Reachable only from the gateway.** No host port is published for this
   service (mirrors every other model lane — see
   ``tests/test_associate_exposure.py``'s ``TestNoUnauthenticatedGeneratePort``),
   and the ``INNEREYE_BASE_URL`` the gateway dials is the compose-network
   name, ``http://comfyui:8188`` (already wired by t9/t11 — this task does not
   touch that plumbing, only proves the service it targets exists and matches).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[1]
_FLEET_COMPOSE = _REPO / "lobes" / "templates" / "fleet" / "docker-compose.yml"


def _fleet_text() -> str:
    return _FLEET_COMPOSE.read_text(encoding="utf-8")


def _fleet() -> dict:
    return yaml.safe_load(_fleet_text())


def _service() -> dict:
    return _fleet()["services"]["comfyui"]


def _service_block_text() -> str:
    """The raw (un-parsed) YAML text of just the ``comfyui:`` service block,
    comments included -- needed for the rationale-comment assertions that a
    parsed dict would silently discard."""
    text = _fleet_text()
    start = text.index("\n  comfyui:\n")
    # The block ends at the next top-level (2-space-indented) service key.
    rest = text[start + 1 :]
    match = re.search(r"\n  [a-z][a-z0-9-]*:\s*\n", rest[1:])
    end = start + 2 + match.start() if match else len(text)
    return text[start:end]


# ---------------------------------------------------------------------------
# criterion 1 -- expose-only isolation
# ---------------------------------------------------------------------------


class TestExposeOnlyIsolation:
    def test_service_exists(self) -> None:
        assert "comfyui" in _fleet()["services"]

    def test_no_ports_key_only_expose(self) -> None:
        svc = _service()
        assert "ports" not in svc, "the comfyui service must not publish a host port"
        assert [str(p) for p in svc.get("expose", [])] == ["8188"]

    def test_gated_behind_the_innereye_compose_profile(self) -> None:
        svc = _service()
        assert svc.get("profiles") == ["innereye"]

    def test_binds_0_0_0_0_not_loopback(self) -> None:
        svc = _service()
        command = svc.get("command")
        assert command is not None
        flattened = " ".join(str(tok) for tok in command) if isinstance(command, list) else command
        assert "--listen" in flattened
        assert "0.0.0.0" in flattened
        assert "127.0.0.1" not in flattened

    def test_port_argument_is_8188(self) -> None:
        svc = _service()
        command = svc["command"]
        flattened = " ".join(str(tok) for tok in command) if isinstance(command, list) else command
        assert "--port" in flattened
        assert "8188" in flattened

    def test_no_host_networking(self) -> None:
        assert _service().get("network_mode") != "host"

    def test_only_the_gateway_publishes_a_host_port(self) -> None:
        """Same invariant test_associate_exposure.py pins for the generate
        lanes: adding comfyui must not create a second published port."""
        services = _fleet()["services"]
        publishing = sorted(name for name, svc in services.items() if svc.get("ports"))
        assert publishing == ["gateway"]

    def test_rationale_comment_names_c36_and_explains_no_ports_key(self) -> None:
        block = _service_block_text()
        assert "c36" in block, "the rationale comment must cite innereye finding c36"
        assert "127.0.0.1" in block, "the comment must name what it is NOT copying"
        assert "ports" in block
        assert "comfyui:8188" in block or "http://comfyui" in block


# ---------------------------------------------------------------------------
# criterion 2 -- a bespoke healthcheck, not the shared /health literal
# ---------------------------------------------------------------------------


class TestBespokeHealthcheck:
    def test_healthcheck_present(self) -> None:
        assert "healthcheck" in _service()

    def test_probes_object_info_or_root_not_health(self) -> None:
        test = _service()["healthcheck"]["test"]
        joined = " ".join(test)
        assert "/health" not in joined, "ComfyUI has no /health route"
        assert ("/object_info" in joined) or joined.rstrip().endswith("8188/")
        assert "curl" in joined

    def test_start_period_and_interval_are_declared(self) -> None:
        """A start_period that gives the app time to finish initializing
        before the first probe is what makes "unhealthy during the loading
        window, not immediately healthy" observable at all."""
        hc = _service()["healthcheck"]
        assert hc.get("start_period")
        assert hc.get("interval")
        assert hc.get("retries")

    def test_healthcheck_comment_explains_the_loading_window(self) -> None:
        block = _service_block_text()
        assert "no /health" in block or "no ``/health``" in block or "/health" in block
        assert "unhealthy" in block.lower()

    def test_shared_health_literal_in_runtime_health_is_untouched(self) -> None:
        """AC2's surviving obligation: lobes/runtime/_health.py's CLI-side
        probe literal must not be parameterized by this task."""
        text = (_REPO / "lobes" / "runtime" / "_health.py").read_text(encoding="utf-8")
        assert "/health" in text

    def test_gateway_readiness_per_backend_path_is_untouched(self) -> None:
        """t11 already made the gateway-side readiness probe per-backend and
        wired /object_info for the render lane; this task must not undo it."""
        text = (_REPO / "lobes" / "gateway" / "_readiness.py").read_text(encoding="utf-8")
        assert "health_path" in text


# ---------------------------------------------------------------------------
# criterion 3 -- reachable only from the gateway, on the compose network
# ---------------------------------------------------------------------------


class TestReachableOnlyFromGateway:
    def test_no_service_uses_host_networking(self) -> None:
        for name, svc in _fleet()["services"].items():
            assert svc.get("network_mode") != "host", f"{name} must not use host networking"

    def test_innereye_base_url_targets_the_compose_service_name(self) -> None:
        """The gateway env passthrough (t9) + shape render (t11) already wire
        this; comfyui must be the target they point at."""
        text = _fleet_text()
        assert "INNEREYE_BASE_URL=${INNEREYE_BASE_URL:-}" in text
        # shape_render.py (not compose) supplies the actual default
        # http://comfyui:8188 -- confirm that constant still matches this
        # service's name/port.
        render_text = (_REPO / "lobes" / "profiles" / "shape_render.py").read_text(encoding="utf-8")
        assert "http://comfyui:8188" in render_text


# ---------------------------------------------------------------------------
# reused fleet shapes (instruction: verbatim reuse, not bespoke invention)
# ---------------------------------------------------------------------------


class TestReusesFleetShapesVerbatim:
    def test_restart_unless_stopped(self) -> None:
        assert _service().get("restart") == "unless-stopped"

    def test_gpu_reservation_matches_the_shared_shape(self) -> None:
        devices = _service()["deploy"]["resources"]["reservations"]["devices"]
        assert devices == [{"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}]

    def test_env_file_chains_env_then_secrets(self) -> None:
        env_file = _service()["env_file"]
        paths = [entry["path"] for entry in env_file]
        assert paths == [".env", ".secrets.env"]
        assert all(entry.get("required") is False for entry in env_file)

    def test_mg_logwrap_entrypoint(self) -> None:
        assert _service()["entrypoint"] == ["bash", "/usr/local/bin/mg-logwrap"]
        volumes = _service()["volumes"]
        assert any("mg-logwrap.sh:/usr/local/bin/mg-logwrap:ro" in v for v in volumes)

    def test_builds_from_the_t2_dockerfile(self) -> None:
        build = _service()["build"]
        assert build["dockerfile"] == "Dockerfile.comfyui"

    def test_volumes_include_models_and_output_and_a_user_key(self) -> None:
        """t4 adds the models/output bind mounts and a `user:` override --
        see tests/test_comfyui_volumes.py for the full t4 acceptance-criteria
        coverage. This is just the scope-boundary flip from t3's own test."""
        svc = _service()
        assert "user" in svc
        volumes = svc.get("volumes", [])
        assert any("/opt/ComfyUI/models" in v for v in volumes)
        assert any("/opt/ComfyUI/output" in v for v in volumes)


# ---------------------------------------------------------------------------
# the FLEET_TEMPLATES wiring the t2 agent flagged as required the moment a
# compose service references Dockerfile.comfyui
# ---------------------------------------------------------------------------


def test_dockerfile_comfyui_is_scaffolded_alongside_compose() -> None:
    from lobes.runtime import _compose

    assert _compose.FLEET_TEMPLATES.get("fleet/Dockerfile.comfyui") == "Dockerfile.comfyui"


def test_comfyui_is_a_gpu_service() -> None:
    from lobes.runtime import _compose

    assert "comfyui" in _compose.GPU_SERVICES
