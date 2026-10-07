"""The ``orin-embed`` deployment shape (orin-embedding-specialist plan, t14).

DECLARED, NOT VALIDATED (#108): no box has booted this shape; every lane budget
is a declaration that plan t15 measures. These tests pin the data, the render
(COMPOSE_PROFILES, per-lane wiring, knobs), the overlay scaffold, and that a
shape with no lanes renders exactly as before.
"""

from __future__ import annotations

from importlib.resources import files

import pytest

from lobes.cli import main
from lobes.embed_lanes import EMBED_LANES
from lobes.profiles.loader import builtin_names, resolve_profile
from lobes.profiles.shape_render import overcommitted_groups, shape_env, shape_services
from lobes.profiles.shapes import AUDIO_ROLES, builtin_shape_names, resolve_shape
from lobes.runtime import _compose, _detect, _env
from tests.goldens.regen import shape_golden_path

_SHAPE = "orin-embed"
_CARD = "orin"
_HOSTED = ("gemma2-embed",)
_NOT_HOSTED = ("qwen3vl-embed", "qwen3vl-rerank", "nemotron-embed", "nomic-code-embed")


def _key(lane: str, suffix: str) -> str:
    return f"{lane.upper().replace('-', '_')}_{suffix}"


def _fake_card(resolved: str) -> _detect.DetectedCard:
    return _detect.DetectedCard(
        resolved=resolved,
        device_name="NVIDIA Test",
        compute_capability="sm_87",
        total_memory_gb=61.3,
        hostname="test-host",
        device_tree_model=None,
        sources={},
    )


def test_hosts_pooling_gears_and_the_specialist_lanes_but_not_associate() -> None:
    shape = resolve_shape(_SHAPE)
    assert set(shape.hosts) == {"embedder", "reranker"}
    for role in ("associate", "cortex", "senses", "hand", *AUDIO_ROLES):
        assert not shape.hosts_role(role)
    assert shape.lanes == _HOSTED
    assert {lane.name for lane in EMBED_LANES} >= set(_HOSTED + _NOT_HOSTED)


def test_measured_budgets_and_nemotron_is_documented_opt_in() -> None:
    """Deviation d1: EG2 is the only standard lane.

    Nemotron's measured knobs are documented, not rendered.
    """
    shape = resolve_shape(_SHAPE)
    assert dict(shape.lane_knobs["gemma2-embed"]) == {
        "mem_limit": "6g",
        "tested_on": "jetson-agx-orin 2026-10-07",
    }
    text = files("lobes.profiles.builtin_shapes").joinpath("orin-embed.toml").read_text("utf-8")
    assert "DECLARED — to be measured" not in text
    assert "NEMOTRON_EMBED_GPU_MEM_UTIL=0.35" in text and "#296" in text
    assert "UNVALIDATED" not in shape.summary


def test_the_overcommit_check_passes() -> None:
    assert not overcommitted_groups(resolve_shape(_SHAPE), resolve_profile(_CARD))


@pytest.mark.parametrize("card", builtin_names())
def test_render_enables_hosted_lanes_and_wires_only_them(card: str) -> None:
    env = shape_env(resolve_shape(_SHAPE), resolve_profile(card))
    profiles = env["COMPOSE_PROFILES"].split(",")
    for lane in _HOSTED:
        assert lane in profiles
        assert env[_key(lane, "BASE_URL")] == f"http://embed-{lane}:8000"
        assert env[_key(lane, "TESTED_ON")] == "jetson-agx-orin 2026-10-07"
    for lane in _NOT_HOSTED:
        assert lane not in profiles
        assert _key(lane, "BASE_URL") not in env
    assert env["GEMMA2_EMBED_MEM_LIMIT"] == "6g"
    assert not any(k.startswith(("QWEN3VL_", "NEMOTRON_", "NOMIC_")) for k in env)
    assert "GEMMA2_EMBED_GPU_MEM_UTIL" not in env  # the sidecar has no vLLM util


def test_services_do_not_include_associate() -> None:
    services = shape_services(resolve_shape(_SHAPE), resolve_profile(_CARD))
    assert "vllm-associate" not in services
    assert "vllm-embed" in services and "vllm-rerank" in services


def test_shapes_without_lanes_render_no_lane_keys() -> None:
    for name in builtin_shape_names():
        if name == _SHAPE:
            continue
        shape = resolve_shape(name)
        assert shape.lanes == ()
        env = shape_env(shape, resolve_profile(_CARD))
        assert not [k for k in env if "EMBED_BASE_URL" in k or k.startswith("QWEN3VL_")]


def test_unknown_lane_or_stray_knob_is_a_load_error() -> None:
    from lobes.cli._errors import ModelGearError
    from lobes.profiles.shapes import Shape

    with pytest.raises(ModelGearError):
        Shape.from_dict("x", {"lanes": ["nope"]})
    with pytest.raises(ModelGearError):
        Shape.from_dict("x", {"lanes": ["gemma2-embed"], "lane_knobs": {"qwen3vl-embed": {}}})
    with pytest.raises(ModelGearError):
        Shape.from_dict(
            "x", {"lanes": ["gemma2-embed"], "lane_knobs": {"gemma2-embed": {"bogus": 1}}}
        )


@pytest.mark.parametrize("card", builtin_names())
def test_goldens_exist_for_every_card(card: str) -> None:
    assert shape_golden_path(_SHAPE, card).is_file()


def test_init_scaffolds_the_embed_overlay_and_renders_the_lanes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(_detect, "detect_card", lambda: _fake_card(_CARD))
    assert main(["init", str(tmp_path), "--profile", _CARD, "--shape", _SHAPE, "--apply"]) == 0
    assert (tmp_path / "docker-compose.embed.yml").is_file()
    assert (tmp_path / "Dockerfile.embed-st").is_file()
    env_text = (tmp_path / _compose.ENV_FILE).read_text(encoding="utf-8")
    assert "GEMMA2_EMBED_BASE_URL=http://embed-gemma2-embed:8000" in env_text
    assert "QWEN3VL_EMBED_BASE_URL" not in env_text
    written = _env.read_env_file(tmp_path / _compose.ENV_FILE)
    assert set(_HOSTED) <= set(written["COMPOSE_PROFILES"].split(","))
    assert _compose.embed_overlay_present(tmp_path)
    args = _compose.compose_file_args(
        audio=False, shape=True, local=False, embed=_compose.embed_overlay_present(tmp_path)
    )
    assert "docker-compose.embed.yml" in args


def test_a_shape_without_lanes_does_not_scaffold_the_overlay(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(_detect, "detect_card", lambda: _fake_card(_CARD))
    assert (
        main(["init", str(tmp_path), "--profile", _CARD, "--shape", "orin-associate", "--apply"])
        == 0
    )
    assert not (tmp_path / "docker-compose.embed.yml").exists()
