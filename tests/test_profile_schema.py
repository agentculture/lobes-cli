"""The new per-role Profile schema + loader (lobes/profiles/schema.py, loader.py).

Guards: (1) a Profile round-trips load -> serialise -> identical; an unknown
role/knob is a load error, never a silent drop; (2) the two shipped built-ins
(spark, thor) carry the exact values the plan requires — spark mirrors the
fleet compose template byte-for-byte, thor encodes exactly its 4 validated
sm_110 divergences and is otherwise identical to spark; (3) an operator
profile dropped in a deployment dir is discovered and overrides a built-in of
the same name, and nothing here mutates a profile at runtime.
"""

from __future__ import annotations

import dataclasses

import pytest

from lobes import machines, profiles
from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError
from lobes.profiles import loader, schema
from lobes.profiles.schema import KNOB_NAMES, ROLES, Profile, RoleProfile

# --- schema: round-trip + validation ---------------------------------------


def test_roles_and_knob_names_are_the_expected_vocabulary() -> None:
    assert ROLES == ("cortex", "senses", "muse", "embedder", "reranker")
    assert set(KNOB_NAMES) == {
        "gpu_mem_util",
        "max_model_len",
        "quantization",
        "kv_cache_dtype",
        "attention_backend",
        "enforce_eager",
        "max_num_seqs",
    }


def test_role_profile_round_trips_through_dict() -> None:
    rp = RoleProfile(
        feasible=True,
        model="some/model",
        gpu_mem_util=0.5,
        max_model_len=4096,
        quantization="fp8",
        kv_cache_dtype="auto",
        attention_backend="TRITON_ATTN",
        enforce_eager=True,
        max_num_seqs=4,
    )
    again = RoleProfile.from_dict("cortex", rp.to_dict())
    assert again == rp


def test_profile_round_trips_through_dict() -> None:
    p = Profile(
        name="custom",
        summary="a test profile",
        roles={"cortex": RoleProfile(model="x", gpu_mem_util=0.4)},
    )
    again = Profile.from_dict("custom", p.to_dict())
    assert again == p
    assert again.to_dict() == p.to_dict()


def test_profile_from_dict_rejects_unknown_role() -> None:
    with pytest.raises(ModelGearError) as exc:
        Profile.from_dict("bogus", {"roles": {"not_a_role": {"model": "x"}}})
    assert exc.value.code == EXIT_USER_ERROR
    assert "not_a_role" in exc.value.message


def test_profile_from_dict_rejects_unknown_knob() -> None:
    with pytest.raises(ModelGearError) as exc:
        Profile.from_dict("bogus", {"roles": {"cortex": {"not_a_knob": 1}}})
    assert exc.value.code == EXIT_USER_ERROR
    assert "not_a_knob" in exc.value.message


def test_role_profile_from_dict_rejects_string_false_for_feasible() -> None:
    # The bug this guards: `feasible = "false"` is a non-empty STRING, which
    # is truthy in Python — the renderer's `if not rp.feasible` must never
    # see a value like this pass validation and silently flip a role to
    # feasible.
    with pytest.raises(ModelGearError) as exc:
        RoleProfile.from_dict("cortex", {"feasible": "false"})
    assert exc.value.code == EXIT_USER_ERROR
    assert "cortex" in exc.value.message
    assert "feasible" in exc.value.message
    assert "bool" in exc.value.message
    assert "str" in exc.value.message


def test_role_profile_from_dict_rejects_string_false_for_enforce_eager() -> None:
    # Same bug for enforce_eager: a truthy string would render `--enforce-eager`
    # from a TOML value the operator intended as "off".
    with pytest.raises(ModelGearError) as exc:
        RoleProfile.from_dict("reranker", {"enforce_eager": "false"})
    assert exc.value.code == EXIT_USER_ERROR
    assert "reranker" in exc.value.message
    assert "enforce_eager" in exc.value.message


def test_role_profile_from_dict_accepts_none_for_enforce_eager() -> None:
    rp = RoleProfile.from_dict("cortex", {"enforce_eager": None})
    assert rp.enforce_eager is None


def test_role_profile_from_dict_rejects_none_for_feasible() -> None:
    # feasible has no Optional in the schema (default True, never None).
    with pytest.raises(ModelGearError) as exc:
        RoleProfile.from_dict("cortex", {"feasible": None})
    assert exc.value.code == EXIT_USER_ERROR
    assert "feasible" in exc.value.message


def test_role_profile_from_dict_rejects_bool_for_gpu_mem_util() -> None:
    # bool is a subclass of int in Python — must be rejected explicitly for a
    # numeric knob, not silently accepted as 0.0/1.0.
    with pytest.raises(ModelGearError) as exc:
        RoleProfile.from_dict("cortex", {"gpu_mem_util": True})
    assert exc.value.code == EXIT_USER_ERROR
    assert "gpu_mem_util" in exc.value.message
    assert "cortex" in exc.value.message


def test_role_profile_from_dict_accepts_int_and_float_for_gpu_mem_util() -> None:
    assert RoleProfile.from_dict("cortex", {"gpu_mem_util": 1}).gpu_mem_util == 1
    assert RoleProfile.from_dict("cortex", {"gpu_mem_util": 0.3}).gpu_mem_util == 0.3


def test_role_profile_from_dict_rejects_bool_for_max_model_len() -> None:
    with pytest.raises(ModelGearError) as exc:
        RoleProfile.from_dict("cortex", {"max_model_len": False})
    assert exc.value.code == EXIT_USER_ERROR
    assert "max_model_len" in exc.value.message


def test_role_profile_from_dict_rejects_bool_for_max_num_seqs() -> None:
    with pytest.raises(ModelGearError) as exc:
        RoleProfile.from_dict("cortex", {"max_num_seqs": True})
    assert exc.value.code == EXIT_USER_ERROR
    assert "max_num_seqs" in exc.value.message


def test_role_profile_from_dict_rejects_wrong_type_for_string_knobs() -> None:
    for knob in ("model", "quantization", "kv_cache_dtype", "attention_backend"):
        with pytest.raises(ModelGearError) as exc:
            RoleProfile.from_dict("cortex", {knob: 123})
        assert exc.value.code == EXIT_USER_ERROR
        assert knob in exc.value.message


def test_role_profile_from_dict_accepts_none_for_string_knobs() -> None:
    for knob in ("model", "quantization", "kv_cache_dtype", "attention_backend"):
        rp = RoleProfile.from_dict("cortex", {knob: None})
        assert getattr(rp, knob) is None


def test_profile_from_dict_rejects_unknown_top_level_key() -> None:
    with pytest.raises(ModelGearError):
        Profile.from_dict("bogus", {"nope": True})


def test_profile_from_dict_rejects_name_mismatch() -> None:
    with pytest.raises(ModelGearError):
        Profile.from_dict("spark", {"name": "thor", "roles": {}})


def test_profile_roles_mapping_is_read_only() -> None:
    p = Profile(name="x", roles={"cortex": RoleProfile(model="a")})
    with pytest.raises(TypeError):
        p.roles["cortex"] = RoleProfile(model="b")  # type: ignore[index]


def test_profile_and_role_profile_are_frozen() -> None:
    p = Profile(name="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.name = "y"  # type: ignore[misc]
    rp = RoleProfile(model="a")
    with pytest.raises(dataclasses.FrozenInstanceError):
        rp.model = "b"  # type: ignore[misc]


def test_role_of_undeclared_role_is_fully_permissive() -> None:
    p = Profile(name="x", roles={"cortex": RoleProfile(model="a")})
    absent = p.role("senses")
    assert absent == RoleProfile()
    assert "senses" not in p.roles


# --- builtins: spark reproduces the shipped fleet template ------------------


def test_spark_builtin_matches_the_fleet_template_exactly() -> None:
    # Literal values from lobes/templates/fleet/docker-compose.yml — ground truth.
    spark = loader.load_builtin("spark")
    assert spark is not None
    assert spark.name == "spark"

    cortex = spark.role("cortex")
    assert cortex.feasible is True
    assert cortex.model == "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"
    assert cortex.gpu_mem_util == 0.30
    assert cortex.max_model_len == 131072
    assert cortex.quantization == "modelopt"
    assert cortex.kv_cache_dtype == "fp8"
    assert cortex.max_num_seqs == 2

    senses = spark.role("senses")
    assert senses.feasible is True
    assert senses.model == "coolthor/gemma-4-12B-it-NVFP4A16"
    assert senses.gpu_mem_util == 0.14
    assert senses.max_model_len == 32768
    assert senses.quantization == "compressed-tensors"
    assert senses.attention_backend == "TRITON_ATTN"

    embedder = spark.role("embedder")
    assert embedder.feasible is True
    assert embedder.model == "Qwen/Qwen3-Embedding-0.6B"
    assert embedder.gpu_mem_util == 0.06
    assert embedder.max_model_len == 8192

    reranker = spark.role("reranker")
    assert reranker.feasible is True
    assert reranker.model == "Qwen/Qwen3-Reranker-0.6B"
    assert reranker.gpu_mem_util == 0.06
    assert reranker.max_model_len == 8192


def test_spark_builtin_round_trips() -> None:
    spark = loader.load_builtin("spark")
    again = Profile.from_dict("spark", spark.to_dict())
    assert again == spark


# --- builtins: thor encodes exactly the 4 validated divergences -------------


def test_thor_builtin_encodes_exactly_the_four_validated_divergences() -> None:
    spark = loader.load_builtin("spark")
    thor = loader.load_builtin("thor")
    assert thor is not None
    assert thor.name == "thor"

    # cortex: only kv_cache_dtype diverges (fp8 -> auto).
    assert thor.role("cortex") == dataclasses.replace(spark.role("cortex"), kv_cache_dtype="auto")

    # senses: identical to spark (no thor divergence declared for this role).
    assert thor.role("senses") == spark.role("senses")

    # embedder: only attention_backend diverges (None -> TRITON_ATTN).
    assert thor.role("embedder") == dataclasses.replace(
        spark.role("embedder"), attention_backend="TRITON_ATTN"
    )

    # reranker: attention_backend + enforce_eager diverge.
    assert thor.role("reranker") == dataclasses.replace(
        spark.role("reranker"), attention_backend="TRITON_ATTN", enforce_eager=True
    )


def test_thor_builtin_divergent_knobs_are_single_sourced_from_machines_registry() -> None:
    # The 4 divergent VALUES are not re-typed in builtin/thor.toml — they come
    # from lobes.machines' thor CardStrategy / SM_110 trait. Prove the two never
    # drift apart: the loaded profile always matches whatever the registry says,
    # even if the registry's provenance strings/values change later.
    thor = loader.load_builtin("thor")
    strategy = machines.get("thor")
    assert strategy is not None
    for role, knobs in strategy.role_knobs().items():
        role_profile = thor.role(role)
        for knob_name, knob in knobs.items():
            assert getattr(role_profile, knob_name) == knob.value


def test_thor_builtin_round_trips() -> None:
    thor = loader.load_builtin("thor")
    again = Profile.from_dict("thor", thor.to_dict())
    assert again == thor


def test_loading_builtins_never_mutates_the_shared_machines_registry() -> None:
    before = machines.get("thor").render()
    loader.load_builtin("thor")
    loader.load_builtin("thor")
    after = machines.get("thor").render()
    assert before == after


# --- loader: resolution + operator overrides --------------------------------


def test_builtin_names_lists_spark_and_thor() -> None:
    names = loader.builtin_names()
    assert "spark" in names
    assert "thor" in names


def test_load_builtin_unknown_name_returns_none() -> None:
    assert loader.load_builtin("does-not-exist") is None


def test_resolve_profile_unknown_name_raises_user_error() -> None:
    with pytest.raises(ModelGearError) as exc:
        loader.resolve_profile("does-not-exist")
    assert exc.value.code == EXIT_USER_ERROR


def test_resolve_profile_resolves_builtin_by_explicit_name() -> None:
    p = loader.resolve_profile("spark")
    assert p.name == "spark"
    p2 = loader.resolve_profile(" THOR ")  # trimmed + lowered
    assert p2.name == "thor"


def test_operator_profile_discovered_in_deployment_dir(tmp_path) -> None:
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "workstation.toml").write_text(
        'name = "workstation"\nsummary = "hand-authored"\n\n'
        '[roles.cortex]\nmodel = "custom/model"\ngpu_mem_util = 0.5\n',
        encoding="utf-8",
    )
    found = loader.discover_operator_profiles(tmp_path)
    assert set(found.keys()) == {"workstation"}
    assert found["workstation"].role("cortex").model == "custom/model"

    resolved = loader.resolve_profile("workstation", deploy_dir=tmp_path)
    assert resolved.role("cortex").gpu_mem_util == 0.5


def test_operator_profile_overrides_a_builtin_of_the_same_name(tmp_path) -> None:
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "spark.toml").write_text(
        'name = "spark"\nsummary = "operator override"\n\n'
        '[roles.cortex]\nmodel = "operator/override-model"\n',
        encoding="utf-8",
    )
    resolved = loader.resolve_profile("spark", deploy_dir=tmp_path)
    assert resolved.role("cortex").model == "operator/override-model"
    # Silent on gpu_mem_util -> "no opinion", NOT the shadowed built-in's 0.30.
    assert resolved.role("cortex").gpu_mem_util is None

    # The built-in itself is never touched by the override.
    builtin_spark = loader.load_builtin("spark")
    assert builtin_spark.role("cortex").model == "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"


def test_mixed_case_operator_file_overrides_the_builtin(tmp_path) -> None:
    # The bug this guards: discover_operator_profiles() used to key profiles
    # by the RAW filename stem, so `profiles/Thor.toml` never matched a
    # resolve_profile("thor") lookup (which normalises with .strip().lower())
    # and silently failed to override the builtin.
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "Thor.toml").write_text(
        '[roles.cortex]\nmodel = "operator/mixed-case-override"\n',
        encoding="utf-8",
    )
    found = loader.discover_operator_profiles(tmp_path)
    assert set(found.keys()) == {"thor"}
    assert found["thor"].role("cortex").model == "operator/mixed-case-override"

    resolved = loader.resolve_profile("thor", deploy_dir=tmp_path)
    assert resolved.role("cortex").model == "operator/mixed-case-override"

    resolved_upper = loader.resolve_profile("THOR", deploy_dir=tmp_path)
    assert resolved_upper.role("cortex").model == "operator/mixed-case-override"


def test_operator_profile_case_collision_raises_user_error(tmp_path) -> None:
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "Thor.toml").write_text('[roles.cortex]\nmodel = "a"\n', encoding="utf-8")
    (profiles_dir / "thor.toml").write_text('[roles.cortex]\nmodel = "b"\n', encoding="utf-8")
    with pytest.raises(ModelGearError) as exc:
        loader.discover_operator_profiles(tmp_path)
    assert exc.value.code == EXIT_USER_ERROR
    assert "thor" in exc.value.message.lower()


def test_discover_operator_profiles_missing_dir_returns_empty(tmp_path) -> None:
    assert loader.discover_operator_profiles(tmp_path / "nonexistent") == {}


def test_available_profiles_merges_builtins_and_operator_with_operator_precedence(
    tmp_path,
) -> None:
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "custom.toml").write_text('name = "custom"\n', encoding="utf-8")
    merged = loader.available_profiles(tmp_path)
    assert "spark" in merged
    assert "thor" in merged
    assert "custom" in merged


def test_malformed_operator_toml_raises_user_error(tmp_path) -> None:
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "broken.toml").write_text("this is not [ valid toml", encoding="utf-8")
    with pytest.raises(ModelGearError) as exc:
        loader.discover_operator_profiles(tmp_path)
    assert exc.value.code == EXIT_USER_ERROR


def test_resolve_profile_does_not_mutate_anything_across_repeated_calls() -> None:
    first = loader.resolve_profile("spark")
    second = loader.resolve_profile("spark")
    assert first == second
    assert first is not second  # independent objects, not a shared mutable singleton


# --- lobes.profiles re-exports the new surface at the package level ---------


def test_profiles_package_reexports_the_new_schema_and_loader_api() -> None:
    assert profiles.Profile is Profile
    assert profiles.RoleProfile is RoleProfile
    assert profiles.ROLES == ROLES
    assert profiles.resolve_profile("spark").name == "spark"
    assert "spark" in profiles.builtin_names()


def test_schema_module_is_importable_directly() -> None:
    assert schema.Profile is Profile
