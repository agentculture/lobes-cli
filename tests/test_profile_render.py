"""Tests for :mod:`lobes.profiles.render` — the Profile -> .env mapping (t4).

Exercises the pure ``profile_env`` function against the two shipped built-ins
(spark, thor, via ``resolve_profile``) and against hand-built ``Profile``/
``RoleProfile`` objects that isolate one behavior at a time (the ``model`` ->
two-keys special case, the ``enforce_eager`` bool -> flag-token translation,
and the ``feasible=False`` marker).
"""

from __future__ import annotations

from lobes.profiles.loader import resolve_profile
from lobes.profiles.render import ROLE_ENV_PREFIX, profile_env
from lobes.profiles.schema import Profile, RoleProfile

# --- role -> prefix table ----------------------------------------------------


def test_role_env_prefix_covers_all_four_roles() -> None:
    assert ROLE_ENV_PREFIX == {
        "cortex": "PRIMARY",
        "senses": "MULTIMODAL",
        "embedder": "EMBED",
        "reranker": "RERANK",
    }


# --- built-in profiles: spot-check the real mapping -------------------------


def test_spark_profile_env_matches_compose_defaults() -> None:
    spark = resolve_profile("spark")
    env = profile_env(spark)
    assert env["PRIMARY_MODEL"] == "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"
    assert env["PRIMARY_SERVED_NAME"] == "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"
    assert env["PRIMARY_GPU_MEM_UTIL"] == "0.3"
    assert env["PRIMARY_MAX_MODEL_LEN"] == "131072"
    assert env["PRIMARY_QUANTIZATION"] == "modelopt"
    assert env["PRIMARY_KV_CACHE_DTYPE"] == "fp8"
    assert env["PRIMARY_MAX_NUM_SEQS"] == "2"
    assert env["MULTIMODAL_MODEL"] == "coolthor/gemma-4-12B-it-NVFP4A16"
    assert env["MULTIMODAL_ATTENTION_BACKEND"] == "TRITON_ATTN"
    assert env["EMBED_MODEL"] == "Qwen/Qwen3-Embedding-0.6B"
    assert env["RERANK_MODEL"] == "Qwen/Qwen3-Reranker-0.6B"
    # No feasibility markers — every role is feasible=True on spark.
    assert not any(k.endswith("_FEASIBLE") for k in env)


def test_thor_profile_env_carries_machine_derived_divergences() -> None:
    thor = resolve_profile("thor")
    env = profile_env(thor)
    # The 4 machine-registry-derived divergences (loader._apply_machine_registry).
    assert env["PRIMARY_KV_CACHE_DTYPE"] == "auto"
    assert env["EMBED_ATTENTION_BACKEND"] == "TRITON_ATTN"
    assert env["RERANK_ATTENTION_BACKEND"] == "TRITON_ATTN"
    assert env["RERANK_ENFORCE_EAGER"] == "--enforce-eager"


def test_profile_env_is_a_dict_of_str_to_str() -> None:
    env = profile_env(resolve_profile("spark"))
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in env.items())


# --- silence: a profile with no opinion on a knob emits nothing -------------


def test_role_with_no_declared_knobs_emits_nothing() -> None:
    profile = Profile(name="bare", roles={})
    assert profile_env(profile) == {}


def test_only_declared_knobs_produce_entries() -> None:
    profile = Profile(
        name="partial",
        roles={"cortex": RoleProfile(gpu_mem_util=0.42)},
    )
    env = profile_env(profile)
    assert env == {"PRIMARY_GPU_MEM_UTIL": "0.42"}


# --- model -> two keys -------------------------------------------------------


def test_model_renders_to_model_and_served_name() -> None:
    profile = Profile(
        name="custom",
        roles={"embedder": RoleProfile(model="acme/my-embedder")},
    )
    env = profile_env(profile)
    assert env == {
        "EMBED_MODEL": "acme/my-embedder",
        "EMBED_SERVED_NAME": "acme/my-embedder",
    }


# --- enforce_eager bool -> flag token ---------------------------------------


def test_enforce_eager_true_renders_the_flag_token() -> None:
    profile = Profile(
        name="eager-on",
        roles={"reranker": RoleProfile(enforce_eager=True)},
    )
    assert profile_env(profile) == {"RERANK_ENFORCE_EAGER": "--enforce-eager"}


def test_enforce_eager_false_renders_the_no_flag_token() -> None:
    profile = Profile(
        name="eager-off",
        roles={"reranker": RoleProfile(enforce_eager=False)},
    )
    assert profile_env(profile) == {"RERANK_ENFORCE_EAGER": "--no-enforce-eager"}


# --- feasible=False -> marker, nothing else ---------------------------------


def test_infeasible_role_renders_only_the_feasible_marker() -> None:
    profile = Profile(
        name="no-senses",
        roles={"senses": RoleProfile(feasible=False, model="would-not-be-served")},
    )
    env = profile_env(profile)
    assert env == {"MULTIMODAL_FEASIBLE": "false"}


def test_feasible_true_role_has_no_feasible_key() -> None:
    profile = Profile(
        name="feasible-true",
        roles={"cortex": RoleProfile(feasible=True, gpu_mem_util=0.5)},
    )
    env = profile_env(profile)
    assert "PRIMARY_FEASIBLE" not in env
    assert env == {"PRIMARY_GPU_MEM_UTIL": "0.5"}


# --- all four roles independently addressable --------------------------------


def test_all_four_roles_map_to_distinct_prefixes() -> None:
    profile = Profile(
        name="all-roles",
        roles={
            "cortex": RoleProfile(gpu_mem_util=0.1),
            "senses": RoleProfile(gpu_mem_util=0.2),
            "embedder": RoleProfile(gpu_mem_util=0.3),
            "reranker": RoleProfile(gpu_mem_util=0.4),
        },
    )
    env = profile_env(profile)
    assert env == {
        "PRIMARY_GPU_MEM_UTIL": "0.1",
        "MULTIMODAL_GPU_MEM_UTIL": "0.2",
        "EMBED_GPU_MEM_UTIL": "0.3",
        "RERANK_GPU_MEM_UTIL": "0.4",
    }
