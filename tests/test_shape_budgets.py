"""Shape-aware budget re-derivation (brain-shapes t2, issue #113).

A shape that DROPS a lobe re-derives the remaining lobes' budgets as DECLARED
DATA in the shape's ``overrides`` table — never a runtime mutation. This is
composed over the #108 per-machine :class:`~lobes.profiles.schema.Profile`
(the "co-resident" values a card profile ships today), never re-implementing
it:

* ``spark-lobe`` drops ``senses`` -> ``cortex`` reclaims the freed budget
  (the co-resident spark values: cortex ``gpu_mem_util`` 0.30, senses 0.14 —
  see ``lobes/profiles/builtin/spark.toml``);
* ``thor-lobe`` drops ``cortex`` -> ``senses`` reclaims the freed budget
  (the co-resident thor values are identical to spark's baseline — see
  ``lobes/profiles/builtin/thor.toml``);
* ``machine-as-brain`` drops nothing -> carries ZERO overrides, so its
  composed budgets stay byte-identical to the shipped card-profile values.

An override's ``None`` fields mean "no opinion" (mirroring
:meth:`~lobes.profiles.schema.Profile.role`'s convention, reused verbatim by
:meth:`~lobes.profiles.shapes.Shape.override`) — composing an override onto a
resolved card :class:`Profile` overlays only the NON-``None`` override fields;
this module's ``_compose`` helper is a narrow, test-local reading of that
composition semantics (the real renderer lands in t3), not a claim about how
t3 will implement it.

``max_model_len`` is deliberately left at the co-resident value on both
mesh-lobe shapes here — raising it to the model's full native context
(cortex's 262144 solo-native ceiling) is explicitly out of scope for this
task (issue #112); only ``gpu_mem_util`` is reclaimed.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from lobes.profiles.loader import resolve_profile
from lobes.profiles.schema import RoleProfile
from lobes.profiles.shapes import load_builtin_shape

# --- co-resident baselines, read live from the shipped card profiles --------
# (never hardcoded twice — if builtin/spark.toml or builtin/thor.toml ever
# change these values, these tests re-derive from the same source of truth
# the shape overrides are supposed to be reclaiming FROM.)

_SPARK_PROFILE = resolve_profile("spark")
_THOR_PROFILE = resolve_profile("thor")

_SPARK_CORTEX_UTIL = _SPARK_PROFILE.role("cortex").gpu_mem_util
_SPARK_SENSES_UTIL = _SPARK_PROFILE.role("senses").gpu_mem_util
_SPARK_CORTEX_MAX_LEN = _SPARK_PROFILE.role("cortex").max_model_len

_THOR_CORTEX_UTIL = _THOR_PROFILE.role("cortex").gpu_mem_util
_THOR_SENSES_UTIL = _THOR_PROFILE.role("senses").gpu_mem_util
_THOR_SENSES_MAX_LEN = _THOR_PROFILE.role("senses").max_model_len


def _compose(base: RoleProfile, override: RoleProfile) -> RoleProfile:
    """Overlay a shape's override onto a card profile's role -- test-local only.

    Non-``None`` override fields win; a ``None`` override field means "no
    opinion", so the base (card-profile) value passes through unchanged. Not
    imported from ``lobes`` -- t3 owns the real renderer; this is just enough
    logic for THIS test module to assert composed budgets.
    """
    overlaid = {}
    for f in fields(RoleProfile):
        override_value = getattr(override, f.name)
        overlaid[f.name] = override_value if override_value is not None else getattr(base, f.name)
    return RoleProfile(**overlaid)


# --- acceptance criterion 1: spark-lobe gives cortex a strictly larger budget -


def test_spark_lobe_declares_a_cortex_override() -> None:
    spark_lobe = load_builtin_shape("spark-lobe")
    assert "cortex" in spark_lobe.overrides
    # No other role is touched -- only cortex reclaims senses' dropped budget.
    assert set(spark_lobe.overrides.keys()) == {"cortex"}


def test_spark_lobe_cortex_override_is_strictly_larger_than_co_resident_budget() -> None:
    spark_lobe = load_builtin_shape("spark-lobe")
    override = spark_lobe.override("cortex")
    composed = _compose(_SPARK_PROFILE.role("cortex"), override)

    # The acceptance criterion, verbatim: util > 0.30 OR max-model-len > 131072.
    assert composed.gpu_mem_util > _SPARK_CORTEX_UTIL or composed.max_model_len > (
        _SPARK_CORTEX_MAX_LEN
    )


def test_spark_lobe_cortex_util_reclaims_exactly_the_dropped_senses_budget() -> None:
    # Provenance, not just "some number bigger than 0.30": cortex's util rises
    # by EXACTLY the dropped senses lobe's co-resident util share.
    override = load_builtin_shape("spark-lobe").override("cortex")
    assert override.gpu_mem_util == pytest.approx(_SPARK_CORTEX_UTIL + _SPARK_SENSES_UTIL)
    assert override.gpu_mem_util == pytest.approx(0.44)


def test_spark_lobe_cortex_max_model_len_stays_at_co_resident_value() -> None:
    # #112 scope, not this task -- the override must NOT touch max_model_len.
    override = load_builtin_shape("spark-lobe").override("cortex")
    assert override.max_model_len is None
    composed = _compose(_SPARK_PROFILE.role("cortex"), override)
    assert composed.max_model_len == _SPARK_CORTEX_MAX_LEN == 131072


# --- thor-lobe: senses reclaims the dropped cortex's budget, symmetrically ---


def test_thor_lobe_declares_a_senses_override() -> None:
    thor_lobe = load_builtin_shape("thor-lobe")
    assert "senses" in thor_lobe.overrides
    assert set(thor_lobe.overrides.keys()) == {"senses"}


def test_thor_lobe_senses_override_is_strictly_larger_than_co_resident_budget() -> None:
    thor_lobe = load_builtin_shape("thor-lobe")
    override = thor_lobe.override("senses")
    composed = _compose(_THOR_PROFILE.role("senses"), override)

    assert composed.gpu_mem_util > _THOR_SENSES_UTIL or composed.max_model_len > (
        _THOR_SENSES_MAX_LEN
    )


def test_thor_lobe_senses_util_reclaims_exactly_the_dropped_cortex_budget() -> None:
    override = load_builtin_shape("thor-lobe").override("senses")
    assert override.gpu_mem_util == pytest.approx(_THOR_SENSES_UTIL + _THOR_CORTEX_UTIL)
    assert override.gpu_mem_util == pytest.approx(0.44)


def test_thor_lobe_senses_max_model_len_stays_at_co_resident_value() -> None:
    override = load_builtin_shape("thor-lobe").override("senses")
    assert override.max_model_len is None
    composed = _compose(_THOR_PROFILE.role("senses"), override)
    assert composed.max_model_len == _THOR_SENSES_MAX_LEN == 32768


# --- acceptance criterion 2: machine-as-brain is untouched, byte-identical ---


def test_machine_as_brain_declares_no_overrides() -> None:
    mab = load_builtin_shape("machine-as-brain")
    assert dict(mab.overrides) == {}


def test_machine_as_brain_composed_budgets_equal_card_profile_exactly() -> None:
    # Zero overrides -> composing changes NOTHING: the composed budget for
    # every role machine-as-brain hosts is exactly the card profile's own
    # value, on both validated cards -- "byte-identical to today's shipped
    # values" restated as a composition fact rather than a file hash.
    mab = load_builtin_shape("machine-as-brain")
    for profile in (_SPARK_PROFILE, _THOR_PROFILE):
        for role in ("cortex", "senses", "embedder", "reranker"):
            base = profile.role(role)
            composed = _compose(base, mab.override(role))
            assert composed == base


def test_machine_as_brain_toml_source_has_no_overrides_table() -> None:
    # Belt-and-suspenders on the "declares no overrides" contract: the raw
    # TOML itself must never grow an [overrides.*] TABLE, not just happen to
    # parse to an empty one. (The bare word "overrides" legitimately appears
    # in this file's prose comments explaining the contract -- only the
    # actual TOML table header is disallowed.)
    from importlib.resources import files

    text = (
        files("lobes.profiles.builtin_shapes")
        .joinpath("machine-as-brain.toml")
        .read_text(encoding="utf-8")
    )
    assert "[overrides." not in text
    assert "\noverrides" not in text
    assert not text.startswith("overrides")


# --- every override carries provenance -----------------------------------


@pytest.mark.parametrize("shape_name", ["spark-lobe", "thor-lobe"])
def test_shape_toml_source_names_its_override_provenance(shape_name: str) -> None:
    # Acceptance criterion: "every derived value carries a provenance comment
    # in the TOML naming its cause" -- checked against the raw source text
    # (Shape.from_dict/RoleProfile don't preserve comments, so this can only
    # be verified by reading the file), looking for the reclaim-from-dropped
    # story in prose near the override.
    from importlib.resources import files

    text = (
        files("lobes.profiles.builtin_shapes")
        .joinpath(f"{shape_name}.toml")
        .read_text(encoding="utf-8")
    )
    assert "[overrides." in text
    assert "reclaim" in text.lower()
    assert "dropped" in text.lower()


# --- differ from machine-as-brain only by hosts + this one reclaim ----------


def test_spark_lobe_and_thor_lobe_overrides_are_the_only_non_hosts_divergence() -> None:
    mab = load_builtin_shape("machine-as-brain")
    spark_lobe = load_builtin_shape("spark-lobe")
    thor_lobe = load_builtin_shape("thor-lobe")

    assert dict(mab.overrides) == {}
    assert set(spark_lobe.overrides.keys()) == {"cortex"}
    assert set(thor_lobe.overrides.keys()) == {"senses"}

    # Neither mesh-lobe shape touches a role's `model`/`feasible`/other knobs --
    # only gpu_mem_util is reclaimed (the acceptance criterion's exact ask).
    for shape, role in (("spark_lobe", "cortex"), ("thor_lobe", "senses")):
        override = (spark_lobe if shape == "spark_lobe" else thor_lobe).override(role)
        assert override.model is None
        assert override.feasible is True
        assert override.quantization is None
        assert override.kv_cache_dtype is None
        assert override.attention_backend is None
        assert override.enforce_eager is None
        assert override.max_num_seqs is None
