"""The Shape schema + built-in shapes (lobes/profiles/shapes.py) — brain-shapes t1.

A :class:`~lobes.profiles.shapes.Shape` declares the ROLE SUBSET a box hosts —
the deployment-shape axis, orthogonal to the #108 per-machine :class:`Profile`
(which says how each hosted role is TUNED on a given card). Guards:

(1) a Shape round-trips load -> serialise -> identical; an unknown role
    anywhere in a shape (``hosts`` or ``overrides``) is a LOAD ERROR, never a
    silently dropped key;
(2) the four built-in shapes (``machine-as-brain``, ``spark-lobe``,
    ``thor-lobe``, ``orin-small``) are expressible as data files with ZERO
    per-shape Python forks — ``spark-lobe``/``thor-lobe`` differ from
    ``machine-as-brain`` only by role subset (``hosts``) and budget
    overrides (``overrides``); ``orin-small`` (mesh-brain end-state t2,
    issue #112) hosts the opt-in ``minor`` gear instead of either heavy
    lobe, declared-but-UNVALIDATED per the #108 rule;
(3) stt/tts are first-class shape members (the audio-overlay pair) alongside
    the four Profile-machinery core roles, and ``minor`` is a hostable-but-
    opt-in fourth kind of role — an unknown role is still a load error
    either way.
"""

from __future__ import annotations

import dataclasses

import pytest

from lobes import profiles
from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError
from lobes.profiles.schema import KNOB_NAMES
from lobes.profiles.schema import ROLES as PROFILE_ROLES
from lobes.profiles.schema import RoleProfile
from lobes.profiles.shapes import (
    AUDIO_ROLES,
    COLLEAGUE_ROLES,
    DEFAULT_HOSTED_ROLES,
    OPT_IN_CORE_ROLES,
    OPT_IN_ROLES,
    SHAPE_ROLES,
    Shape,
    builtin_shape_names,
    load_builtin_shape,
    resolve_shape,
)

# --- vocabulary --------------------------------------------------------------


def test_audio_roles_are_stt_and_tts() -> None:
    assert AUDIO_ROLES == ("stt", "tts")


def test_opt_in_roles_is_minor() -> None:
    # `minor` (mesh-brain end-state t2, issue #112): the opt-in vllm-minor
    # generate gear, added to the Shape schema's hostable vocabulary WITHOUT
    # joining the six first-class Colleague roles — see CLAUDE.md's
    # "Colleague roles" section ("the 4B minor ... are opt-in gears and not
    # first-class Colleague roles").
    assert OPT_IN_ROLES == ("minor",)


def test_colleague_roles_is_profile_roles_plus_audio_roles() -> None:
    # The seven first-class Colleague roles (issue #81): the five Profile-
    # machinery core roles plus the two audio-overlay sidecars. No role
    # vocabulary is re-typed here — it is composed from schema.ROLES. This is
    # the Colleague CONTRACT set; the set machine-as-brain hosts is the
    # narrower DEFAULT_HOSTED_ROLES (opt-in core roles excluded — see
    # test_machine_as_brain_hosts_every_default_role below).
    assert COLLEAGUE_ROLES == PROFILE_ROLES + AUDIO_ROLES
    assert COLLEAGUE_ROLES == ("cortex", "senses", "muse", "embedder", "reranker", "stt", "tts")


def test_default_hosted_roles_is_colleague_roles_minus_opt_in_core() -> None:
    # The machine-as-brain identity set: every Colleague role EXCEPT the
    # opt-in core lobes (muse — a 31B that cannot co-reside with the
    # cortex+senses duo; hosted only by an explicit muse-hosting shape).
    assert OPT_IN_CORE_ROLES == ("muse",)
    assert DEFAULT_HOSTED_ROLES == tuple(
        role for role in COLLEAGUE_ROLES if role not in OPT_IN_CORE_ROLES
    )
    assert DEFAULT_HOSTED_ROLES == ("cortex", "senses", "embedder", "reranker", "stt", "tts")


def test_shape_roles_is_colleague_roles_plus_opt_in_roles() -> None:
    # SHAPE_ROLES (everything a shape may declare hosted) is a strict
    # superset of COLLEAGUE_ROLES (everything machine-as-brain hosts): the
    # opt-in `minor` gear is hostable by a shape (e.g. orin-small) but is
    # never part of "every role this card can serve".
    assert SHAPE_ROLES == COLLEAGUE_ROLES + OPT_IN_ROLES
    assert SHAPE_ROLES == (
        "cortex",
        "senses",
        "muse",
        "embedder",
        "reranker",
        "stt",
        "tts",
        "minor",
    )


# --- Shape: round-trip + validation ------------------------------------------


def test_shape_round_trips_through_dict() -> None:
    s = Shape(
        name="custom",
        summary="a test shape",
        hosts=("cortex", "embedder", "stt"),
        overrides={"cortex": RoleProfile(gpu_mem_util=0.5, max_model_len=200000)},
    )
    again = Shape.from_dict("custom", s.to_dict())
    assert again == s
    assert again.to_dict() == s.to_dict()


def test_shape_from_dict_defaults_to_empty_hosts_and_overrides() -> None:
    s = Shape.from_dict("bare", {})
    assert s.name == "bare"
    assert s.hosts == ()
    assert dict(s.overrides) == {}


def test_shape_from_dict_rejects_unknown_role_in_hosts() -> None:
    with pytest.raises(ModelGearError) as exc:
        Shape.from_dict("bogus", {"hosts": ["not_a_role"]})
    assert exc.value.code == EXIT_USER_ERROR
    assert "not_a_role" in exc.value.message


def test_shape_from_dict_rejects_unknown_role_in_overrides() -> None:
    with pytest.raises(ModelGearError) as exc:
        Shape.from_dict("bogus", {"overrides": {"not_a_role": {"gpu_mem_util": 0.5}}})
    assert exc.value.code == EXIT_USER_ERROR
    assert "not_a_role" in exc.value.message


def test_shape_from_dict_rejects_audio_role_in_overrides() -> None:
    # stt/tts have no machine-dependent vLLM knobs of their own (they map onto
    # the fixed audio-overlay sidecars, not the Profile/RoleProfile machinery)
    # -- an override entry for one is a load error, not a silent no-op.
    with pytest.raises(ModelGearError) as exc:
        Shape.from_dict("bogus", {"overrides": {"stt": {"gpu_mem_util": 0.5}}})
    assert exc.value.code == EXIT_USER_ERROR
    assert "stt" in exc.value.message


def test_shape_from_dict_rejects_minor_role_in_overrides() -> None:
    # `minor` likewise carries no Profile/RoleProfile knobs of its own (its
    # budget is the compose template's own fixed defaults) -- an override
    # entry for it is a load error, exactly like stt/tts above.
    with pytest.raises(ModelGearError) as exc:
        Shape.from_dict("bogus", {"overrides": {"minor": {"gpu_mem_util": 0.5}}})
    assert exc.value.code == EXIT_USER_ERROR
    assert "minor" in exc.value.message


def test_shape_from_dict_rejects_unknown_top_level_key() -> None:
    with pytest.raises(ModelGearError) as exc:
        Shape.from_dict("bogus", {"nope": True})
    assert exc.value.code == EXIT_USER_ERROR
    assert "nope" in exc.value.message


def test_shape_from_dict_rejects_name_mismatch() -> None:
    with pytest.raises(ModelGearError):
        Shape.from_dict("spark-lobe", {"name": "thor-lobe"})


def test_shape_from_dict_reuses_role_profile_validation_for_overrides() -> None:
    # Composed over the #108 schema, not re-implemented: an override's knob
    # validation is exactly RoleProfile.from_dict's (e.g. a truthy STRING for
    # a numeric knob is rejected, matching test_profile_schema.py's guard).
    with pytest.raises(ModelGearError) as exc:
        Shape.from_dict("bogus", {"overrides": {"cortex": {"gpu_mem_util": "false"}}})
    assert exc.value.code == EXIT_USER_ERROR
    assert "gpu_mem_util" in exc.value.message
    assert "cortex" in exc.value.message


def test_shape_overrides_mapping_is_read_only() -> None:
    s = Shape(name="x", hosts=("cortex",), overrides={"cortex": RoleProfile(model="a")})
    with pytest.raises(TypeError):
        s.overrides["cortex"] = RoleProfile(model="b")  # type: ignore[index]


def test_shape_is_frozen() -> None:
    s = Shape(name="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.name = "y"  # type: ignore[misc]


def test_shape_hosts_role() -> None:
    s = Shape(name="x", hosts=("cortex", "stt"))
    assert s.hosts_role("cortex") is True
    assert s.hosts_role("senses") is False


def test_shape_override_of_undeclared_role_is_fully_permissive() -> None:
    s = Shape(name="x", hosts=("cortex",), overrides={"cortex": RoleProfile(model="a")})
    assert s.override("senses") == RoleProfile()
    assert "senses" not in s.overrides


# --- built-ins: the four shapes are expressible as pure data -----------------


def test_builtin_shape_names_lists_all_four() -> None:
    names = builtin_shape_names()
    assert set(names) == {
        "machine-as-brain",
        "spark-lobe",
        "thor-lobe",
        "thor-muse",
        "orin-small",
    }


def test_machine_as_brain_hosts_every_default_role() -> None:
    mab = load_builtin_shape("machine-as-brain")
    assert mab is not None
    assert mab.name == "machine-as-brain"
    # machine-as-brain hosts the six DEFAULT-hosted Colleague roles -- NOT
    # the opt-in `minor` gear, and NOT the opt-in core `muse` lobe: both are
    # deliberately excluded from "every role this card can serve" (see
    # DEFAULT_HOSTED_ROLES vs COLLEAGUE_ROLES/SHAPE_ROLES above).
    assert set(mab.hosts) == set(DEFAULT_HOSTED_ROLES)
    assert "minor" not in mab.hosts
    assert "muse" not in mab.hosts


def test_machine_as_brain_carries_no_overrides() -> None:
    mab = load_builtin_shape("machine-as-brain")
    assert dict(mab.overrides) == {}


def test_spark_lobe_hosts_cortex_embedder_reranker_and_audio_no_senses() -> None:
    spark_lobe = load_builtin_shape("spark-lobe")
    assert spark_lobe is not None
    assert set(spark_lobe.hosts) == {"cortex", "embedder", "reranker", "stt", "tts"}
    assert "senses" not in spark_lobe.hosts


def test_thor_lobe_hosts_senses_embedder_reranker_and_audio_no_cortex() -> None:
    thor_lobe = load_builtin_shape("thor-lobe")
    assert thor_lobe is not None
    assert set(thor_lobe.hosts) == {"senses", "embedder", "reranker", "stt", "tts"}
    assert "cortex" not in thor_lobe.hosts


def test_thor_muse_hosts_muse_pooling_and_audio_no_default_heavy_lobe() -> None:
    # The muse reference shape: hosts the opt-in 31B muse lobe + the cheap
    # gears, and drops BOTH heavy default lobes (cortex to a spark-lobe box,
    # senses to a peer box, e.g. an Orin — declared via MULTIMODAL_PEER_*).
    thor_muse = load_builtin_shape("thor-muse")
    assert thor_muse is not None
    assert set(thor_muse.hosts) == {"muse", "embedder", "reranker", "stt", "tts"}
    assert "cortex" not in thor_muse.hosts
    assert "senses" not in thor_muse.hosts
    # The FULL muse declaration lives in the shape's overrides (the card
    # profiles stay silent): model + budget hypothesis + backend divergence.
    muse = thor_muse.override("muse")
    assert muse.model == "nvidia/Gemma-4-31B-IT-NVFP4"
    assert muse.gpu_mem_util == 0.40
    assert muse.max_model_len == 262144
    assert muse.quantization == "modelopt"
    assert muse.attention_backend == "TRITON_ATTN"


def test_orin_small_hosts_minor_embedder_reranker_and_audio_no_heavy_lobe() -> None:
    # mesh-brain end-state t2 (issue #112): the Jetson AGX Orin 64GB
    # reference shape hosts NEITHER heavy generate lobe -- cortex and senses
    # are both absent -- and hosts the opt-in `minor` gear instead.
    orin_small = load_builtin_shape("orin-small")
    assert orin_small is not None
    assert set(orin_small.hosts) == {"minor", "embedder", "reranker", "stt", "tts"}
    assert "cortex" not in orin_small.hosts
    assert "senses" not in orin_small.hosts


def test_orin_small_carries_no_overrides() -> None:
    # `minor` carries no Profile knobs to re-derive, and embedder/reranker
    # are not reclaiming anything a dropped role freed (nothing co-resided
    # with them before) -- orin-small declares zero overrides, like
    # machine-as-brain.
    orin_small = load_builtin_shape("orin-small")
    assert dict(orin_small.overrides) == {}


def test_spark_lobe_and_thor_lobe_carry_reclaimed_budget_overrides() -> None:
    # t2 (brain-shapes, landed after this file) fills these in as re-derived
    # budget overrides for the lobe that gains the dropped role's freed
    # budget: spark-lobe's cortex reclaims dropped senses' util; thor-lobe's
    # senses reclaims dropped cortex's util. machine-as-brain (nothing
    # dropped) stays empty -- see test_machine_as_brain_carries_no_overrides
    # above. The re-derived VALUES + their provenance are exercised in
    # tests/test_shape_budgets.py, not repeated here.
    assert set(load_builtin_shape("spark-lobe").overrides.keys()) == {"cortex"}
    assert set(load_builtin_shape("thor-lobe").overrides.keys()) == {"senses"}


def test_spark_lobe_and_thor_lobe_differ_from_machine_as_brain_only_by_hosts_and_overrides() -> (
    None
):
    mab = load_builtin_shape("machine-as-brain")
    spark_lobe = load_builtin_shape("spark-lobe")
    thor_lobe = load_builtin_shape("thor-lobe")

    # spark-lobe drops exactly senses; adds nothing beyond machine-as-brain's set.
    assert set(mab.hosts) - set(spark_lobe.hosts) == {"senses"}
    assert set(spark_lobe.hosts) - set(mab.hosts) == set()

    # thor-lobe drops exactly cortex; adds nothing beyond machine-as-brain's set.
    assert set(mab.hosts) - set(thor_lobe.hosts) == {"cortex"}
    assert set(thor_lobe.hosts) - set(mab.hosts) == set()

    # machine-as-brain (nothing dropped) carries no override; spark-lobe/
    # thor-lobe (brain-shapes t2) each carry exactly one -- the lobe that
    # reclaims the dropped role's freed budget. The full re-derived VALUES
    # (with provenance) are exercised in tests/test_shape_budgets.py.
    assert dict(mab.overrides) == {}
    assert set(spark_lobe.overrides.keys()) == {"cortex"}
    assert set(thor_lobe.overrides.keys()) == {"senses"}


def test_builtin_shapes_round_trip() -> None:
    for name in ("machine-as-brain", "spark-lobe", "thor-lobe", "orin-small"):
        shape = load_builtin_shape(name)
        again = Shape.from_dict(name, shape.to_dict())
        assert again == shape


# --- loader: resolution -------------------------------------------------------


def test_load_builtin_shape_unknown_name_returns_none() -> None:
    assert load_builtin_shape("does-not-exist") is None


def test_resolve_shape_unknown_name_raises_user_error() -> None:
    with pytest.raises(ModelGearError) as exc:
        resolve_shape("does-not-exist")
    assert exc.value.code == EXIT_USER_ERROR


def test_resolve_shape_resolves_builtin_by_explicit_name() -> None:
    s = resolve_shape("spark-lobe")
    assert s.name == "spark-lobe"
    s2 = resolve_shape(" THOR-LOBE ")  # trimmed + lowered, matching resolve_profile's convention
    assert s2.name == "thor-lobe"


# --- genericity: zero per-shape Python forks ---------------------------------


def test_shape_schema_is_fully_generic_no_hardcoded_shape_names() -> None:
    # A brand-new, never-seen-before shape name loads and round-trips with the
    # exact same code path as the three built-ins -- proof the schema carries
    # no per-shape-name branching.
    data = {
        "summary": "a hypothetical fourth shape",
        "hosts": ["senses", "tts"],
        "overrides": {"senses": {"max_model_len": 16384}},
    }
    shape = Shape.from_dict("hypothetical-lobe", data)
    again = Shape.from_dict("hypothetical-lobe", shape.to_dict())
    assert again == shape
    assert set(shape.hosts) == {"senses", "tts"}
    assert shape.override("senses").max_model_len == 16384


def test_shape_knob_names_available_for_overrides_match_profile_schema() -> None:
    # Confirms overrides are the SAME knob vocabulary as RoleProfile -- not a
    # parallel, re-typed set of override fields.
    rp = RoleProfile.from_dict("cortex", {name: None for name in KNOB_NAMES})
    shape = Shape(name="x", hosts=("cortex",), overrides={"cortex": rp})
    assert shape.override("cortex") == rp


# --- lobes.profiles re-exports the shape schema + loader ---------------------


def test_profiles_package_reexports_shape_schema_and_loader() -> None:
    assert profiles.Shape is Shape
    assert profiles.SHAPE_ROLES == SHAPE_ROLES
    assert profiles.AUDIO_ROLES == AUDIO_ROLES
    assert profiles.COLLEAGUE_ROLES == COLLEAGUE_ROLES
    assert profiles.OPT_IN_ROLES == OPT_IN_ROLES
    assert profiles.resolve_shape("spark-lobe").name == "spark-lobe"
    assert "machine-as-brain" in profiles.builtin_shape_names()
    assert "orin-small" in profiles.builtin_shape_names()
