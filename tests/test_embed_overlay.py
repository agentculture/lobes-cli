"""The specialist embed/rerank lane overlay (orin-embedding-specialist, t13; o12)."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import yaml

import lobes.gateway  # noqa: F401  (resolves the lobes.embed_lanes <-> roles import cycle)
from lobes.embed_lanes import EMBED_LANES
from lobes.runtime import _compose

_FLEET = files("lobes.templates").joinpath("fleet")


def _load(name: str) -> dict:
    return yaml.safe_load(_FLEET.joinpath(name).read_text(encoding="utf-8"))


def _services() -> dict:
    return _load("docker-compose.embed.yml")["services"]


def test_every_registry_lane_has_a_service_named_after_it() -> None:
    services = _services()
    for lane in EMBED_LANES:
        svc = services[f"embed-{lane.name}"]
        assert svc["container_name"] == f"embed-{lane.name}"
        assert svc["profiles"] == [lane.name]


def test_no_specialist_service_publishes_a_host_port() -> None:  # o12
    for lane in EMBED_LANES:
        svc = _services()[f"embed-{lane.name}"]
        assert "ports" not in svc, lane.name
        assert svc["expose"] == ["8000"]


def test_every_specialist_service_has_mem_limit_and_healthcheck() -> None:  # o12
    for lane in EMBED_LANES:
        svc = _services()[f"embed-{lane.name}"]
        assert svc.get("mem_limit"), lane.name
        assert svc["healthcheck"]["test"], lane.name
        assert svc["healthcheck"]["start_period"], lane.name


def test_no_depends_on_edges_anywhere_in_the_overlay() -> None:
    for name, svc in _services().items():
        assert "depends_on" not in svc, name


def test_overlay_only_adds_lane_services_and_the_gateway_env() -> None:
    expected = {f"embed-{lane.name}" for lane in EMBED_LANES} | {"gateway"}
    assert set(_services()) == expected
    assert set(_services()["gateway"]) == {"environment"}


def test_gateway_receives_every_lane_base_url_empty_by_default() -> None:
    env = _services()["gateway"]["environment"]
    for lane in EMBED_LANES:
        assert f"{lane.base_url_env}=${{{lane.base_url_env}:-}}" in env


def test_engine_matches_the_service_shape() -> None:
    for lane in EMBED_LANES:
        svc = _services()[f"embed-{lane.name}"]
        if lane.engine == "vllm":
            cmd = svc["command"]
            assert "--runner=pooling" in cmd
            assert f"--convert={'classify' if lane.task == 'score' else 'embed'}" in cmd
        else:
            assert svc["build"]["dockerfile"] == "Dockerfile.embed-st"


def test_sidecar_dockerfile_carries_the_spike_pins() -> None:
    text = _FLEET.joinpath("Dockerfile.embed-st").read_text(encoding="utf-8")
    assert "sha256:7c5a10e9a8b3c8642f4d0463a41215176c0dd834b4f0967287c7e3e517cf1be9" in text
    assert '"transformers==5.19.0"' in text
    assert '"sentence-transformers==6.1.0"' in text


def test_overlay_and_dockerfile_are_scaffold_templates() -> None:
    assert _compose.EMBED_TEMPLATES == {
        "fleet/docker-compose.embed.yml": _compose.EMBED_OVERLAY,
        "fleet/Dockerfile.embed-st": "Dockerfile.embed-st",
    }
    for src in _compose.EMBED_TEMPLATES:
        assert files("lobes.templates").joinpath(src).is_file()


def test_base_fleet_compose_does_not_declare_lane_services() -> None:
    base = _load("docker-compose.yml")["services"]
    assert not [s for s in base if s.startswith("embed-")]


def test_chain_adds_embed_overlay_only_when_requested() -> None:
    assert _compose.compose_file_args(audio=False, shape=False, local=False) == []
    chain = _compose.compose_file_args(audio=False, shape=False, local=False, embed=True)
    assert chain == ["-f", _compose.COMPOSE_FILE, "-f", _compose.EMBED_OVERLAY]


def test_chain_orders_embed_after_audio_before_shape_and_override() -> None:
    names = [
        t
        for t in _compose.compose_file_args(
            audio=True, audio_he=True, shape=True, local=True, gpu=True, embed=True
        )
        if t != "-f"
    ]
    assert names.index(_compose.AUDIO_HE_OVERLAY) < names.index(_compose.EMBED_OVERLAY)
    assert names.index(_compose.EMBED_OVERLAY) < names.index(_compose.SHAPE_OVERLAY)
    assert names.index(_compose.SHAPE_OVERLAY) < names.index(_compose.LOCAL_OVERRIDE)


def test_compose_files_probes_the_deploy_dir_for_the_overlay(tmp_path: Path) -> None:
    assert _compose._compose_files(tmp_path) == []
    (tmp_path / _compose.EMBED_OVERLAY).write_text("services: {}\n")
    assert _compose.embed_overlay_present(tmp_path)
    assert _compose._compose_files(tmp_path) == [
        "-f",
        _compose.COMPOSE_FILE,
        "-f",
        _compose.EMBED_OVERLAY,
    ]


def test_gateway_receives_every_lane_key_the_gateway_reads() -> None:
    """The gateway does not read .env: every per-lane key it consumes must be passed through."""
    from lobes.gateway._config import lane_env_key

    env = _load("docker-compose.embed.yml")["services"]["gateway"]["environment"]
    names = {item.split("=", 1)[0] for item in env}
    for lane in EMBED_LANES:
        for suffix in ("BASE_URL", "FEASIBLE", "MAX_ACTIVE", "TESTED_ON", "MAX_MODEL_LEN"):
            assert lane_env_key(lane.name, suffix) in names


def test_sidecar_default_modalities_are_accepted_by_the_sidecar() -> None:
    """The overlay's EMBED_MODALITIES default must parse.

    "code" is text to the sidecar, not a modality.
    """
    import re

    from lobes.embed_sidecar.server import parse_modalities

    env = _load("docker-compose.embed.yml")["services"]["embed-gemma2-embed"]["environment"]
    line = next(item for item in env if item.startswith("EMBED_MODALITIES="))
    default = re.search(r":-([^}]*)}", line).group(1)
    assert parse_modalities(default) == frozenset({"text", "image", "video", "audio"})
