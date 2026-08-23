"""The ``orin-cortex`` deployment shape — the Orin as a LOCAL cortex host (t5).

The Jetson AGX Orin has always reached ``cortex`` through a peer: NVFP4 exports
quantize ACTIVATIONS to FP4 and need Blackwell tensor cores, and sm_87 is
Ampere. That line is about the CHECKPOINT FORMAT, not the role — a weight-only
GGUF served by ``llama-server`` decodes on Ampere fine — so the card now
declares ``cortex`` feasible on the catalog's first non-vLLM gear, and this
shape is what hosts it.

Three things have to line up for that to be more than a declaration, and this
module asserts all three:

1. **The right lane runs.** ``cortex``'s compose service is engine-aware
   (:func:`lobes.profiles.shape_render.role_service`): a GGUF gear resolves to
   ``llamacpp-primary``, never ``vllm-primary`` — which could not load a
   ``.gguf`` at all.
2. **The other lane does NOT.** ``lobes init`` parks ``vllm-primary`` in the
   inert ``shape-dropped`` compose profile even though the shape HOSTS cortex,
   because the role is hosted by the other engine. Both cortex lanes running at
   once on a 61.3 GiB board is the failure this prevents.
3. **senses is dropped, honestly.** The two do not fit together (MEASURED:
   ~33 GiB + ~27.6 GiB against 61.3 GiB with ZERO swap), so this shape gives up
   vision on this box and flags it off rather than half-serving it.

**HONESTY (#108).** ``orin-cortex`` is DECLARED, UNVALIDATED — no box has
booted this shape and no acceptance transcript exists (that is the covering
plan's t10). The t1 spike that measured the numbers behind it returned
"functional GO / throughput FAIL-AS-SPECIFIED": correct decode at the full
window, and 2.61 tok/s, BELOW the plan's >= 5 tok/s gate. Nothing in this module
asserts a validation claim — only that the declared data renders what it says.
"""

from __future__ import annotations

from importlib.resources import files

import pytest

from lobes.catalog import ENGINE_LLAMA_CPP
from lobes.cli._commands import init as init_cmd
from lobes.profiles.loader import builtin_names, resolve_profile
from lobes.profiles.render import (
    LLAMA_CPP_ACTIVATION_ENV,
    LLAMA_CPP_COMPOSE_PROFILE,
    ROLE_ENV_PREFIX,
    role_engine,
)
from lobes.profiles.shape_render import (
    LLAMA_CPP_ROLE_SERVICE,
    ROLE_SERVICE,
    render_shape,
    shape_env,
    shape_services,
)
from lobes.profiles.shapes import AUDIO_ROLES, resolve_shape
from tests.goldens.regen import shape_golden_path

_SHAPE = "orin-cortex"
_CARD = "orin"
_GGUF_ID = "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"


def _shape_toml(name: str) -> str:
    return (
        files("lobes.profiles.builtin_shapes").joinpath(f"{name}.toml").read_text(encoding="utf-8")
    )


# --- 1. the shape's own data -------------------------------------------------


def test_hosts_cortex_hand_and_the_pooling_gears_and_drops_senses() -> None:
    shape = resolve_shape(_SHAPE)
    assert set(shape.hosts) == {"cortex", "hand", "embedder", "reranker"}
    assert not shape.hosts_role("senses")
    for role in AUDIO_ROLES:
        assert not shape.hosts_role(role), "sm_87 has no Parakeet image (see orin-lobe)"


def test_declares_no_overrides_because_there_is_nothing_to_reclaim() -> None:
    """Unlike spark-lobe/thor-lobe, dropping a lobe frees no budget to spend here.

    ``llama-server`` has no ``gpu_memory_utilization`` knob at all — its
    footprint is weights (fixed by the quantization inside the ``.gguf``) plus
    KV (fixed by ``-c``) — and the context is already at the checkpoint's own
    native ceiling. The card profile's declaration therefore stands unmodified.
    """
    assert dict(resolve_shape(_SHAPE).overrides) == {}
    text = _shape_toml(_SHAPE)
    assert "[overrides." not in text
    assert "RECLAIMS NOTHING" in text


def test_the_co_residency_measurement_is_recorded_not_computed() -> None:
    """The reason senses goes is a MEASURED pair of numbers, cited in the file."""
    text = _shape_toml(_SHAPE)
    assert "MEASURED" in text
    assert "33 GiB" in text  # the llama.cpp cortex, measured at the served window
    assert "61.3 GiB" in text  # the board
    assert "ZERO swap" in text  # the board has none — nothing absorbs an overshoot


def test_claims_no_validation() -> None:
    """#108: no box has booted this shape, so the file must not imply one has."""
    text = _shape_toml(_SHAPE)
    assert "DECLARED, NOT VALIDATED" in text
    assert "VALIDATED on" not in text
    assert "UNVALIDATED" in resolve_shape(_SHAPE).summary


# --- 2. the engine-aware render ---------------------------------------------


def test_cortex_runs_the_llama_cpp_lane_and_not_the_vllm_one() -> None:
    profile = resolve_profile(_CARD)
    assert role_engine(profile.role("cortex")) == ENGINE_LLAMA_CPP
    services = shape_services(resolve_shape(_SHAPE), profile)
    assert LLAMA_CPP_ROLE_SERVICE["cortex"] in services
    assert ROLE_SERVICE["cortex"] not in services
    assert ROLE_SERVICE["senses"] not in services
    for role in ("hand", "embedder", "reranker"):
        assert ROLE_SERVICE[role] in services


def test_the_render_names_the_gear_and_wires_its_lane_in_one_go() -> None:
    env = shape_env(resolve_shape(_SHAPE), resolve_profile(_CARD))
    assert env["PRIMARY_MODEL"] == _GGUF_ID
    # The gateway routes off PRIMARY_SERVED_NAME; a mismatch is a 404 behind a
    # healthy backend, so the two must agree.
    assert env["PRIMARY_SERVED_NAME"] == _GGUF_ID
    assert env["PRIMARY_URL"] == LLAMA_CPP_ACTIVATION_ENV["cortex"]["PRIMARY_URL"]
    assert env["COMPOSE_PROFILES"].split(",") == [LLAMA_CPP_COMPOSE_PROFILE]
    # The full native window the spike served, and NO utilization fraction —
    # this engine has no such flag, so declaring one would be a dead knob (#92).
    assert env["PRIMARY_MAX_MODEL_LEN"] == "262144"
    assert "PRIMARY_GPU_MEM_UTIL" not in env
    assert "PRIMARY_QUANTIZATION" not in env


def test_dropped_senses_is_flagged_off_and_leaks_no_knob() -> None:
    rendered = render_shape(resolve_shape(_SHAPE), resolve_profile(_CARD))
    prefix = ROLE_ENV_PREFIX["senses"]
    assert rendered.env.get(f"{prefix}_FEASIBLE") == "false"
    leaked = [k for k in rendered.env if k.startswith(f"{prefix}_") and k != f"{prefix}_FEASIBLE"]
    assert leaked == [], f"dropped senses leaked knob env: {leaked}"


def test_the_card_keeps_its_tegra_iowait_declaration_under_this_shape() -> None:
    """Card facts survive every shape — the whole point of ``host_env``."""
    env = shape_env(resolve_shape(_SHAPE), resolve_profile(_CARD))
    assert env["LOBES_IOWAIT_DEGRADED_THRESHOLD"] == "100"


@pytest.mark.parametrize("card", builtin_names())
def test_goldens_exist_for_every_card(card: str) -> None:
    assert shape_golden_path(_SHAPE, card).is_file()


# --- 3. init parks the OTHER engine's cortex lane ---------------------------


def test_init_parks_the_vllm_cortex_lane_even_though_cortex_is_hosted() -> None:
    """The engine-swap half of the drop decision.

    ``vllm-primary`` is unconditional in the base fleet template, so without
    this it would start alongside ``llamacpp-primary`` and crash-loop trying to
    load a ``.gguf`` — while holding memory the running lane needs.
    """
    dropped = init_cmd._shape_dropped_services(resolve_shape(_SHAPE), resolve_profile(_CARD))
    assert dropped == ["vllm-multimodal", "vllm-primary"]
    text = init_cmd.render_shape_override(resolve_shape(_SHAPE), resolve_profile(_CARD))
    assert text is not None
    assert "  vllm-primary:" in text
    assert f'profiles: ["{init_cmd.SHAPE_DROPPED_PROFILE}"]' in text
    # ...and the lane that DOES run is never parked.
    assert LLAMA_CPP_ROLE_SERVICE["cortex"] not in text


def test_the_engine_axis_parks_nothing_extra_on_a_vllm_card() -> None:
    """Narrowness: on every card whose roles are vLLM gears, the parked set is
    exactly what it was before the engine axis — the shape's own drops, nothing
    more. That is the property that keeps every pre-existing deployment's
    generated ``docker-compose.shape.yml`` byte-identical.
    """
    from lobes.profiles.schema import ROLES
    from lobes.profiles.shapes import OPT_IN_CORE_ROLES, builtin_shape_names

    for card in builtin_names():
        profile = resolve_profile(card)
        if any(role_engine(profile.role(r)) == ENGINE_LLAMA_CPP for r in ROLES):
            continue
        for shape_name in builtin_shape_names():
            shape = resolve_shape(shape_name)
            expected = sorted(
                ROLE_SERVICE[role]
                for role in ROLES
                if role not in OPT_IN_CORE_ROLES and not shape.hosts_role(role)
            )
            assert init_cmd._shape_dropped_services(shape, profile) == expected


def test_end_to_end_init_render_needs_no_manual_compose_edit(tmp_path, monkeypatch) -> None:
    """The acceptance criterion, exercised through the real CLI.

    A fresh ``lobes init --profile orin --shape orin-cortex --apply`` must
    produce the GPU-access override (``runtime: nvidia`` — this board's toolkit
    runs csv mode and refuses the ``deploy.resources`` form), the shape override
    that parks the vLLM cortex lane, and the ``.env`` that un-gates and wires
    the llama.cpp one. No hand edit anywhere in that list.
    """
    from lobes.cli import main
    from lobes.runtime import _compose, _detect

    from .test_init_shape import _fake_card

    monkeypatch.setattr(_detect, "detect_card", lambda: _fake_card(_CARD))
    assert main(["init", str(tmp_path), "--profile", _CARD, "--shape", _SHAPE, "--apply"]) == 0

    gpu_override = (tmp_path / _compose.GPU_OVERLAY).read_text(encoding="utf-8")
    assert f"  {LLAMA_CPP_ROLE_SERVICE['cortex']}:" in gpu_override
    assert "runtime: nvidia" in gpu_override
    assert "deploy: !reset null" in gpu_override

    shape_override = (tmp_path / _compose.SHAPE_OVERLAY).read_text(encoding="utf-8")
    assert "  vllm-primary:" in shape_override

    env_text = (tmp_path / _compose.ENV_FILE).read_text(encoding="utf-8")
    assert f"PRIMARY_URL={LLAMA_CPP_ACTIVATION_ENV['cortex']['PRIMARY_URL']}" in env_text
    assert f"COMPOSE_PROFILES={LLAMA_CPP_COMPOSE_PROFILE}" in env_text
    assert f"PRIMARY_MODEL={_GGUF_ID}" in env_text
