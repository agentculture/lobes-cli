"""The workload (purpose) + machine profile tables are a single source of truth —
guard their invariants, the detection markers, and the resolution layering."""

from __future__ import annotations

import pytest

from model_gear import profiles
from model_gear.cli._errors import EXIT_USER_ERROR, ModelGearError

# --- workload profiles ----------------------------------------------------


def test_workload_table_nonempty_and_balanced_is_default() -> None:
    assert len(profiles.WORKLOAD_PROFILES) >= 3
    assert profiles.DEFAULT_PURPOSE == "balanced"
    assert profiles.WORKLOAD_PROFILES[0].name == profiles.DEFAULT_PURPOSE


def test_workload_names_and_aliases_are_unique_and_self_referential() -> None:
    names = [wp.name for wp in profiles.WORKLOAD_PROFILES]
    assert len(names) == len(set(names))
    seen: set[str] = set()
    for wp in profiles.WORKLOAD_PROFILES:
        assert wp.name in wp.aliases, f"{wp.name} not in its own aliases"
        for alias in wp.aliases:
            assert alias not in seen, f"duplicate alias {alias}"
            assert alias == alias.lower(), f"alias {alias} must be lowercase"
            seen.add(alias)


def test_workload_bench_shapes_match_the_report() -> None:
    # The (input, output) shapes mirror shahizat's three workloads.
    shapes = {
        wp.name: (wp.bench_input_len, wp.bench_output_len) for wp in profiles.WORKLOAD_PROFILES
    }
    assert shapes["balanced"] == (1000, 1000)
    assert shapes["prompt-heavy"] == (8000, 1000)
    assert shapes["decode-heavy"] == (1000, 8000)


def test_balanced_batching_matches_the_report() -> None:
    bal = profiles.workload_profile("balanced")
    assert bal.max_num_seqs == 4
    assert bal.max_num_batched_tokens == 8192


def test_workload_profile_resolves_canonical_and_alias() -> None:
    assert profiles.workload_profile("decode-heavy").name == "decode-heavy"
    assert profiles.workload_profile("decode").name == "decode-heavy"
    assert profiles.workload_profile(" Balance ").name == "balanced"  # trimmed + lowered


def test_workload_profile_unknown_raises_user_error() -> None:
    with pytest.raises(ModelGearError) as exc:
        profiles.workload_profile("turbo")
    assert exc.value.code == EXIT_USER_ERROR


# --- machine profiles -----------------------------------------------------


def test_machine_table_has_unique_names_and_a_generic_fallback() -> None:
    names = [mp.name for mp in profiles.MACHINE_PROFILES]
    assert len(names) == len(set(names))
    generic = profiles.machine_profile("generic")
    assert generic.gpu_markers == ()  # never auto-matched


def test_machine_attention_backend_is_always_set() -> None:
    # The template default substitutes ${VLLM_ATTENTION_BACKEND}; an empty value
    # would emit a broken `--attention-backend=` token.
    for mp in profiles.MACHINE_PROFILES:
        assert mp.attention_backend


def test_spark_serves_128k_by_default() -> None:
    # Load-tested 2026-06-03 on the shared GB10: the 256K-native MTP primary serves
    # at 128K (~70 GiB resident at util 0.6, same as 32K — the KV pool holds 9.6x a
    # full 128K request). Guard the shipped default so it can't drift back to the old
    # 32K first-load cap, while util stays conservative (the box is shared).
    spark = profiles.machine_profile("spark")
    assert spark.max_model_len == 131072
    assert spark.gpu_mem_util == 0.6
    # The other machines keep their own contexts — only spark was measured at 128K.
    assert profiles.machine_profile("blackwell").max_model_len == 65536
    assert profiles.machine_profile("thor").max_model_len == 32768
    assert profiles.machine_profile("generic").max_model_len == 32768


def test_machine_profile_unknown_raises_user_error() -> None:
    with pytest.raises(ModelGearError) as exc:
        profiles.machine_profile("h100")
    assert exc.value.code == EXIT_USER_ERROR


def test_detect_machine_markers() -> None:
    assert profiles.detect_machine("NVIDIA GB10", "spark") == "spark"
    assert profiles.detect_machine("unknown", "DGX-Spark") == "spark"
    assert profiles.detect_machine("NVIDIA Thor", "thor-01") == "thor"
    assert profiles.detect_machine("NVIDIA RTX PRO 6000 Blackwell", "ws") == "blackwell"
    assert profiles.detect_machine("some other gpu", "build-box") == "generic"
    assert profiles.detect_machine(None, None) == "generic"


def test_detect_machine_gb10_is_spark_not_blackwell() -> None:
    # The GB10 is itself a Grace *Blackwell* part — it must resolve to spark, and
    # never trip the discrete-Blackwell profile.
    assert profiles.detect_machine("NVIDIA GB10 Grace Blackwell", "spark") == "spark"


def test_resolve_machine_auto_detects_else_passes_through() -> None:
    assert profiles.resolve_machine("auto", gpu_name="NVIDIA GB10", hostname="x") == "spark"
    assert profiles.resolve_machine("", gpu_name="NVIDIA GB10", hostname="x") == "spark"
    assert profiles.resolve_machine("blackwell") == "blackwell"
    with pytest.raises(ModelGearError):
        profiles.resolve_machine("nope")


# --- resolve_serve_config -------------------------------------------------


def test_resolve_serve_config_layers_machine_and_purpose() -> None:
    cfg = profiles.resolve_serve_config("decode-heavy", "blackwell")
    assert cfg["VLLM_PURPOSE"] == "decode-heavy"
    assert cfg["VLLM_MACHINE"] == "blackwell"
    # machine layer
    assert cfg["VLLM_GPU_MEM_UTIL"] == "0.85"
    assert cfg["VLLM_MAX_MODEL_LEN"] == "65536"
    assert cfg["VLLM_ATTENTION_BACKEND"] == "flashinfer"
    # purpose layer
    assert cfg["VLLM_MAX_NUM_SEQS"] == "8"
    assert cfg["VLLM_MAX_NUM_BATCHED_TOKENS"] == "4096"


def test_resolve_serve_config_explicit_overrides_win() -> None:
    cfg = profiles.resolve_serve_config("balanced", "spark", max_model_len=16384, gpu_mem_util=0.5)
    assert cfg["VLLM_MAX_MODEL_LEN"] == "16384"
    assert cfg["VLLM_GPU_MEM_UTIL"] == "0.5"


def test_resolve_serve_config_accepts_purpose_alias() -> None:
    cfg = profiles.resolve_serve_config("prompt", "spark")
    assert cfg["VLLM_PURPOSE"] == "prompt-heavy"  # normalized to canonical


def test_as_dicts_round_trip() -> None:
    assert {d["name"] for d in profiles.workloads_as_dicts()} == {
        wp.name for wp in profiles.WORKLOAD_PROFILES
    }
    assert {d["name"] for d in profiles.machines_as_dicts()} == {
        mp.name for mp in profiles.MACHINE_PROFILES
    }
