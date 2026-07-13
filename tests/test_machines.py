"""The per-chip strategy registry — one module per card, derived legacy surface.

Guards the three things t12 promises: (1) adding a chip is one register() call
that detection / profile resolution / knob rendering all pick up with zero edits
elsewhere; (2) the legacy profiles surface is unchanged for callers; (3) the Thor
strategy carries the load-tested values, and detection still falls back to
generic.
"""

from __future__ import annotations

import pytest

from lobes import machines, profiles
from lobes.machines import (
    CardStrategy,
    DetectionSignature,
    Knob,
    MachineDefaults,
    Trait,
)


@pytest.fixture
def _clean_registry():
    """Snapshot the registry so a test's synthetic chips never leak."""
    before = machines.names()
    yield
    for name in machines.names():
        if name not in before:
            machines.unregister(name)


# --- registry basics ------------------------------------------------------


def test_builtins_registered_in_detection_precedence_order() -> None:
    # spark before blackwell: the GB10 (a Grace *Blackwell* part) must be matched
    # by its specific marker before the discrete Blackwell profile.
    assert machines.names() == ("spark", "thor", "blackwell", "generic")


def test_get_is_honest_none_for_unknown() -> None:
    assert machines.get("spark").name == "spark"
    assert machines.get(" THOR ").name == "thor"  # trimmed + lowered
    assert machines.get("h100") is None  # no silent generic fallback


def test_detect_returns_strategy_or_none_never_generic() -> None:
    assert machines.detect("NVIDIA GB10", "spark").name == "spark"
    assert machines.detect("NVIDIA Thor", "thor-01").name == "thor"
    assert machines.detect("NVIDIA RTX PRO 6000 Blackwell", "ws").name == "blackwell"
    # Honest UNKNOWN: no match -> None (the generic fallback is the caller's job).
    assert machines.detect("some other gpu", "build-box") is None
    assert machines.detect(None, None) is None


def test_generic_is_registered_but_never_auto_detected() -> None:
    assert machines.get("generic") is not None
    assert machines.get("generic").signature.name_markers == ()


# --- thor: the load-tested values -----------------------------------------


def test_thor_is_load_tested_with_sm110_signature() -> None:
    thor = machines.get("thor")
    assert thor.status == "load-tested"
    assert thor.signature.compute_capability == "sm_110"


def test_thor_carries_measured_role_knobs() -> None:
    knobs = machines.get("thor").role_knobs()
    # cortex generate lane
    assert knobs["cortex"]["kv_cache_dtype"].value == "auto"
    # embedder + reranker pooling lanes forced onto Triton (sm_110 quirk)
    assert knobs["embedder"]["attention_backend"].value == "TRITON_ATTN"
    assert knobs["reranker"]["attention_backend"].value == "TRITON_ATTN"
    assert knobs["reranker"]["enforce_eager"].value is True
    # every knob names its cause
    for role_knobs in knobs.values():
        for knob in role_knobs.values():
            assert knob.provenance


def test_thor_legacy_row_no_longer_claims_flashinfer() -> None:
    thor = profiles.machine_profile("thor")
    assert thor.attention_backend != "flashinfer"
    assert thor.attention_backend  # still non-empty (template substitutes it)
    assert thor.status == "load-tested"
    # legacy single-model context stays a sensible 32K (test_profiles locks this)
    assert thor.max_model_len == 32768


def test_pooling_provenance_names_sm110_as_the_cause() -> None:
    knobs = machines.get("thor").role_knobs()
    assert "sm_110" in knobs["embedder"]["attention_backend"].provenance
    assert "sm_110" in knobs["reranker"]["enforce_eager"].provenance


# --- shared trait: sm_110 reused without copy-paste -----------------------


def test_sm110_trait_is_shared_not_copied(_clean_registry) -> None:
    # A second sm_110 board composes the SAME trait — one line, no knob copy — and
    # inherits identical pooling knobs (and their provenance).
    other = CardStrategy(
        name="sm110-clone",
        summary="synthetic second sm_110 board",
        signature=DetectionSignature(name_markers=("clone",), compute_capability="sm_110"),
        defaults=MachineDefaults(
            gpu_mem_util=Knob(0.6, "unified board"),
            max_model_len=Knob(32768, "conservative"),
            attention_backend=Knob("TRITON_ATTN", "sm_110: Triton path"),
        ),
        status="configured",
        traits=(machines.SM_110,),
    )
    machines.register(other)
    assert other.role_knobs()["embedder"] == machines.get("thor").role_knobs()["embedder"]
    assert other.role_knobs()["reranker"]["enforce_eager"].value is True


def test_board_override_beats_trait_knob(_clean_registry) -> None:
    board = CardStrategy(
        name="override-board",
        summary="board that pins one trait knob differently",
        signature=DetectionSignature(name_markers=("ovr",)),
        defaults=MachineDefaults(
            gpu_mem_util=Knob(0.6, "x"),
            max_model_len=Knob(32768, "x"),
            attention_backend=Knob("TRITON_ATTN", "x"),
        ),
        status="configured",
        traits=(machines.SM_110,),
        role_overrides={
            "reranker": {"attention_backend": Knob("FLASHINFER", "board pins its own backend")},
        },
    )
    machines.register(board)
    rk = board.role_knobs()["reranker"]
    assert rk["attention_backend"].value == "FLASHINFER"  # override wins
    assert rk["enforce_eager"].value is True  # trait knob still present


# --- rendering ------------------------------------------------------------


def test_render_is_json_friendly_with_provenance() -> None:
    rendered = machines.get("thor").render()
    assert rendered["name"] == "thor"
    assert rendered["status"] == "load-tested"
    assert rendered["signature"]["compute_capability"] == "sm_110"
    assert rendered["defaults"]["attention_backend"]["value"] == "TRITON_ATTN"
    assert rendered["defaults"]["attention_backend"]["provenance"]
    assert rendered["role_knobs"]["reranker"]["enforce_eager"]["value"] is True


def test_duplicate_registration_rejected_without_replace() -> None:
    spark = machines.get("spark")
    with pytest.raises(ValueError):
        machines.register(spark)


# --- criterion 1: one file + one line, zero edits elsewhere ---------------


def test_new_chip_lights_up_everywhere_with_zero_edits(_clean_registry) -> None:
    """Registering a synthetic chip must flow through detection, profile
    resolution, the legacy MACHINE_PROFILES surface and knob rendering — with no
    edit to profiles.py or machines/__init__.py internals (only this register())."""
    synthetic = CardStrategy(
        name="synthetic",
        summary="a made-up card for the test",
        signature=DetectionSignature(
            name_markers=("madeupchip",), compute_capability="sm_999", total_memory_gb=64
        ),
        defaults=MachineDefaults(
            gpu_mem_util=Knob(0.42, "synthetic"),
            max_model_len=Knob(4096, "synthetic"),
            attention_backend=Knob("TRITON_ATTN", "synthetic"),
        ),
        status="configured",
        role_overrides={"cortex": {"kv_cache_dtype": Knob("fp8", "synthetic")}},
    )
    machines.register(synthetic)

    # detection (via the legacy profiles entry point)
    assert profiles.detect_machine("MadeUpChip GPU", "host") == "synthetic"
    assert machines.detect("madeupchip", None).name == "synthetic"

    # legacy profile resolution + the derived MACHINE_PROFILES tuple
    mp = profiles.machine_profile("synthetic")
    assert mp.gpu_mem_util == 0.42
    assert mp.attention_backend == "TRITON_ATTN"
    assert "synthetic" in {p.name for p in profiles.MACHINE_PROFILES}

    # serve-config layering picks up the synthetic machine defaults
    cfg = profiles.resolve_serve_config("balanced", "synthetic")
    assert cfg["VLLM_MACHINE"] == "synthetic"
    assert cfg["VLLM_MAX_MODEL_LEN"] == "4096"
    assert cfg["VLLM_GPU_MEM_UTIL"] == "0.42"

    # knob rendering
    assert (
        machines.get("synthetic").render()["role_knobs"]["cortex"]["kv_cache_dtype"]["value"]
        == "fp8"
    )


def test_unregister_restores_prior_state(_clean_registry) -> None:
    machines.register(
        CardStrategy(
            name="ephemeral",
            summary="x",
            signature=DetectionSignature(name_markers=("ephem",)),
            defaults=MachineDefaults(
                gpu_mem_util=Knob(0.6, "x"),
                max_model_len=Knob(1024, "x"),
                attention_backend=Knob("TRITON_ATTN", "x"),
            ),
            status="configured",
        )
    )
    assert machines.get("ephemeral") is not None
    machines.unregister("ephemeral")
    assert machines.get("ephemeral") is None
    assert isinstance(machines.SM_110, Trait)
