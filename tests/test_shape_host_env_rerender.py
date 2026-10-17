"""A shape-declared ``[host_env]`` key is rendered by EVERY shape (review finding #5).

``compose_profile`` merges a shape's ``[host_env]`` over the card's, so one
key (``LOBES_BOOT_ORDER`` / ``LOBES_BOOT_STAGGER_S``, declared by
``orin-associate``) can differ between two shapes on the same card. Rendering
only ever WRITES keys, so a shape that stayed silent on the key used to leave the
previous shape's value behind in ``.env``: on the Spark (whose card declares no
``LOBES_BOOT_ORDER``), ``orin-associate`` then ``spark-lobe`` left
``LOBES_BOOT_ORDER=associate`` in place, holding every lane longer and breaking
"byte-for-byte restorable by re-running the previous shape". Every shape now
renders each such key explicitly: the shape's value, else the card's, else the
documented default (docs/contracts/lane-safety-runtime.md section 1).
"""

from __future__ import annotations

import pytest

from lobes.cli import main
from lobes.profiles.loader import builtin_names, resolve_profile
from lobes.profiles.render import SHAPE_HOST_ENV_DEFAULTS, shape_host_env_keys
from lobes.profiles.shape_render import render_shape
from lobes.profiles.shapes import builtin_shape_names, resolve_shape
from lobes.runtime import _detect


def _env(shape: str, card: str) -> dict[str, str]:
    return dict(render_shape(resolve_shape(shape), resolve_profile(card)).env)


def test_the_key_set_is_derived_from_the_builtin_shapes() -> None:
    declared: set[str] = set()
    for name in builtin_shape_names():
        declared |= set(resolve_shape(name).host_env)
    assert shape_host_env_keys() == frozenset(declared)
    assert {"LOBES_BOOT_ORDER", "LOBES_BOOT_STAGGER_S"} <= declared


def test_every_shape_declared_key_has_a_documented_default() -> None:
    # A shape adding a new [host_env] key must also say what every OTHER shape
    # renders for it, or a shape switch could not overwrite it.
    assert shape_host_env_keys() <= set(SHAPE_HOST_ENV_DEFAULTS)
    assert SHAPE_HOST_ENV_DEFAULTS["LOBES_BOOT_ORDER"] == ""
    assert SHAPE_HOST_ENV_DEFAULTS["LOBES_BOOT_STAGGER_S"] == "20"


@pytest.mark.parametrize("card", builtin_names())
@pytest.mark.parametrize("shape", builtin_shape_names())
def test_every_shape_renders_every_shape_declared_key(shape: str, card: str) -> None:
    env = _env(shape, card)
    for key in shape_host_env_keys():
        assert key in env, f"{shape}@{card} does not render {key}"


def test_shape_value_beats_card_value_beats_default() -> None:
    # Shape value (orin-associate declares both keys).
    assert _env("orin-associate", "spark")["LOBES_BOOT_ORDER"] == "associate"
    assert _env("orin-associate", "spark")["LOBES_BOOT_STAGGER_S"] == "400"
    # Card value (the orin card declares a LOBES_BOOT_ORDER; orin-lobe does not).
    assert _env("orin-lobe", "orin")["LOBES_BOOT_ORDER"] == "multimodal,embed,rerank"
    # Documented default (neither the spark card nor spark-lobe declares them).
    assert _env("spark-lobe", "spark")["LOBES_BOOT_ORDER"] == ""
    assert _env("spark-lobe", "spark")["LOBES_BOOT_STAGGER_S"] == "20"
    assert _env("orin-lobe", "orin")["LOBES_BOOT_STAGGER_S"] == "20"


def _spark_card() -> _detect.DetectedCard:
    return _detect.DetectedCard(
        resolved="spark",
        device_name="NVIDIA GB10",
        compute_capability="sm_121",
        total_memory_gb=119.7,
        hostname="test-host",
        device_tree_model=None,
        sources={},
    )


def _switch_spark_lobe_orin_associate_spark_lobe(tmp_path, monkeypatch) -> tuple[str, str, str]:
    monkeypatch.setattr(_detect, "detect_card", _spark_card)
    target = tmp_path / "deploy"
    assert main(["init", "--shape", "spark-lobe", str(target), "--apply"]) == 0
    first = (target / ".env").read_text(encoding="utf-8")
    assert main(["init", "--shape", "orin-associate", str(target), "--apply", "--force"]) == 0
    middle = (target / ".env").read_text(encoding="utf-8")
    assert main(["init", "--shape", "spark-lobe", str(target), "--apply", "--force"]) == 0
    return first, middle, (target / ".env").read_text(encoding="utf-8")


def _host_env_lines(text: str) -> list[str]:
    keys = shape_host_env_keys()
    return [line for line in text.splitlines() if line.split("=", 1)[0] in keys]


def test_switching_away_from_orin_associate_restores_the_host_env_keys(
    tmp_path, monkeypatch
) -> None:
    first, middle, restored = _switch_spark_lobe_orin_associate_spark_lobe(tmp_path, monkeypatch)
    assert "LOBES_BOOT_ORDER=associate\n" in middle  # sanity: it actually changed
    assert "LOBES_BOOT_STAGGER_S=400\n" in middle
    assert "LOBES_BOOT_ORDER=associate" not in restored
    # Rewritten IN PLACE (set_env), never a second appended line.
    assert _host_env_lines(restored) == _host_env_lines(first)
    assert sorted(_host_env_lines(first)) == ["LOBES_BOOT_ORDER=", "LOBES_BOOT_STAGGER_S=20"]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "pre-existing, out of this fix's scope: ROLE keys orin-associate renders and "
        "spark-lobe is silent on (PRIMARY_FEASIBLE=false, ASSOCIATE_*, HAND_FEASIBLE=false, "
        "COMPOSE_PROFILES=associate) are still left behind by a re-render"
    ),
)
def test_switching_away_from_orin_associate_restores_env_byte_for_byte(
    tmp_path, monkeypatch
) -> None:
    first, _, restored = _switch_spark_lobe_orin_associate_spark_lobe(tmp_path, monkeypatch)
    assert restored == first


def test_an_operator_typed_unrelated_key_survives_the_switch(tmp_path, monkeypatch) -> None:
    # Only RENDERED keys are rewritten; a key no shape or card renders is left alone.
    monkeypatch.setattr(_detect, "detect_card", _spark_card)
    target = tmp_path / "deploy"
    assert main(["init", "--shape", "spark-lobe", str(target), "--apply"]) == 0
    env_path = target / ".env"
    env_path.write_text(
        env_path.read_text(encoding="utf-8") + "GATEWAY_API_KEY=operator-typed\n",
        encoding="utf-8",
    )
    assert main(["init", "--shape", "orin-associate", str(target), "--apply", "--force"]) == 0
    assert main(["init", "--shape", "spark-lobe", str(target), "--apply", "--force"]) == 0
    assert "GATEWAY_API_KEY=operator-typed\n" in env_path.read_text(encoding="utf-8")
