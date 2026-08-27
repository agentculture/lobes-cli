"""The new per-role Profile schema + loader (lobes/profiles/schema.py, loader.py).

Guards: (1) a Profile round-trips load -> serialise -> identical; an unknown
role/knob is a load error, never a silent drop; (2) the shipped built-ins
carry the exact values the plan requires — spark mirrors the
fleet compose template byte-for-byte, thor encodes exactly its 4 validated
sm_110 divergences and is otherwise identical to spark, and orin (the one
non-Blackwell card) serves senses on the QAT int4 W4A16 checkpoint while
vetoing every NVFP4 generate lobe; (3) an operator
profile dropped in a deployment dir is discovered and overrides a built-in of
the same name, and nothing here mutates a profile at runtime.
"""

from __future__ import annotations

import dataclasses
from importlib.resources import files

import pytest

from lobes import machines, profiles
from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError
from lobes.machines import orin as machines_orin
from lobes.profiles import loader, schema
from lobes.profiles.schema import (
    KNOB_NAMES,
    ROLES,
    SPECULATIVE_CONFIG_ROLES,
    Profile,
    RoleProfile,
)

# --- schema: round-trip + validation ---------------------------------------


def test_roles_and_knob_names_are_the_expected_vocabulary() -> None:
    assert ROLES == (
        "cortex",
        "senses",
        "muse",
        "worker",
        "associate",
        "hand",
        "embedder",
        "reranker",
    )
    assert set(KNOB_NAMES) == {
        "gpu_mem_util",
        "max_model_len",
        "quantization",
        "kv_cache_dtype",
        "attention_backend",
        "enforce_eager",
        "max_num_seqs",
        "hf_overrides",
        "allow_long_max_model_len",
        "speculative_config",
    }


def test_speculative_config_is_rejected_for_lanes_that_cannot_consume_it() -> None:
    """A knob that cannot take effect must fail LOUDLY at load, not render a dead key.

    Only three compose lanes expand ``<PREFIX>_SPECULATIVE_CONFIG``
    (vllm-primary / vllm-multimodal / vllm-worker). `muse` hardcodes its
    token as a YAML list element, and `hand`/`embedder`/`reranker` carry no
    speculative flag at all — so rendering the key for them would write an
    `.env` variable nothing reads. Raised by review on PR #202 (Qodo finding
    4), which caught exactly that silent no-op for `muse`.
    """
    for role in ("muse", "hand", "embedder", "reranker"):
        with pytest.raises(ModelGearError) as excinfo:
            RoleProfile.from_dict(role, {"speculative_config": "'--speculative-config={}'"})
        assert "speculative_config" in str(excinfo.value)
        # The message must say WHY, not just "no".
        assert "SPECULATIVE_CONFIG" in str(excinfo.value)

    # The empty string is a MEANINGFUL value (spec-decode off), so it must be
    # rejected on those lanes too rather than slipping through as falsy.
    with pytest.raises(ModelGearError):
        RoleProfile.from_dict("muse", {"speculative_config": ""})


def test_speculative_config_is_accepted_for_lanes_that_do_consume_it() -> None:
    for role in sorted(SPECULATIVE_CONFIG_ROLES):
        assert RoleProfile.from_dict(role, {"speculative_config": ""}).speculative_config == ""
        token = '\'--speculative-config={"method":"mtp"}\''
        assert (
            RoleProfile.from_dict(role, {"speculative_config": token}).speculative_config == token
        )


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
        hf_overrides='{"text_config": {"rope_parameters": {"rope_type": "yarn"}}}',
        allow_long_max_model_len="1",
        speculative_config='"\'--speculative-config={\\"method\\":\\"mtp\\"}\'"',
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


# --- host_env: the card-level (non-role) .env table -------------------------


def test_profile_round_trips_with_a_host_env_table() -> None:
    p = Profile(
        name="custom",
        roles={"cortex": RoleProfile(model="x")},
        host_env={"LOBES_IOWAIT_DEGRADED_THRESHOLD": "100"},
    )
    again = Profile.from_dict("custom", p.to_dict())
    assert again == p
    assert again.host_env["LOBES_IOWAIT_DEGRADED_THRESHOLD"] == "100"


def test_profile_without_host_env_defaults_to_empty_and_is_read_only() -> None:
    p = Profile.from_dict("custom", {"roles": {}})
    assert dict(p.host_env) == {}
    with pytest.raises(TypeError):
        p.host_env["X"] = "1"  # type: ignore[index]


def test_profile_from_dict_rejects_non_mapping_host_env() -> None:
    with pytest.raises(ModelGearError) as exc:
        Profile.from_dict("bogus", {"host_env": ["LOBES_X=1"]})
    assert exc.value.code == EXIT_USER_ERROR
    assert "host_env" in exc.value.message


def test_profile_from_dict_rejects_a_non_env_var_name_key() -> None:
    with pytest.raises(ModelGearError) as exc:
        Profile.from_dict("bogus", {"host_env": {"not a var": "1"}})
    assert exc.value.code == EXIT_USER_ERROR
    assert "not a var" in exc.value.message


def test_profile_from_dict_rejects_a_non_string_host_env_value() -> None:
    # Values are written to .env verbatim, so the author must spell the exact
    # bytes -- an int/float would introduce a formatting question (100 vs
    # 100.0) between the TOML and the rendered file.
    with pytest.raises(ModelGearError) as exc:
        Profile.from_dict("bogus", {"host_env": {"LOBES_IOWAIT_DEGRADED_THRESHOLD": 100}})
    assert exc.value.code == EXIT_USER_ERROR
    assert "LOBES_IOWAIT_DEGRADED_THRESHOLD" in exc.value.message
    assert "str" in exc.value.message


def test_profile_from_dict_still_rejects_an_unknown_top_level_key() -> None:
    # host_env widened the known top-level set by exactly one key -- an
    # arbitrary top-level table is still a load error.
    with pytest.raises(ModelGearError) as exc:
        Profile.from_dict("bogus", {"env": {"LOBES_X": "1"}})
    assert exc.value.code == EXIT_USER_ERROR
    assert "env" in exc.value.message


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
    assert cortex.model == "unsloth/Qwen3.8-27B-NVFP4"
    assert cortex.gpu_mem_util == 0.30
    assert cortex.max_model_len == 131072
    assert cortex.quantization == "compressed-tensors"
    assert cortex.kv_cache_dtype == "fp8"
    assert cortex.max_num_seqs == 2
    # The 1M YaRN knobs are spark-lobe-shape-only (t5) -- the bare card
    # profile takes no position on either.
    assert cortex.hf_overrides is None
    assert cortex.allow_long_max_model_len is None

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

    # cortex: kv_cache_dtype diverges (fp8 -> auto) — the one VALIDATED sm_110
    # divergence. The temporary Qwen3.6-vs-3.8 `model` gap the qwen3.8 t5
    # rollout left here closed on 2026-08-20 (deviation d1: a Thor box hosts
    # cortex again and validated the 3.8 checkpoint live at 1M —
    # docs/evidence/2026-08-20-accept-cortex-local-thor.txt), so the model
    # tracks spark.toml again. The Thor-only MTP-off requirement lives in the
    # compose lane, not the schema (see thor.toml's cortex comment).
    assert thor.role("cortex") == dataclasses.replace(
        spark.role("cortex"),
        kv_cache_dtype="auto",
    )

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


# --- builtins: orin serves senses on Ampere and vetoes every NVFP4 lobe -----


def _orin_toml_text() -> str:
    """The packaged builtin/orin.toml source — read for its COMMENTS.

    The knob VALUES are asserted through the loader like every other profile;
    this reads the file itself only where the contract is about what the file
    SAYS (the measured-pending marking), which no loaded Profile can carry.
    """
    return (files(loader.BUILTIN_PACKAGE) / "orin.toml").read_text(encoding="utf-8")


def test_orin_builtin_serves_senses_on_the_qat_w4a16_checkpoint() -> None:
    orin = loader.load_builtin("orin")
    assert orin is not None
    assert orin.name == "orin"

    senses = orin.role("senses")
    assert senses.feasible is True
    # The QAT int4 W4A16 checkpoint — weight-only int4 is what makes an Ampere
    # sm_87 board a candidate at all (the NVFP4 lobes below are not).
    assert senses.model == "unsloth/gemma-4-12B-it-qat-w4a16"
    assert senses.quantization == "compressed-tensors"
    assert senses.attention_backend == "TRITON_ATTN"
    # MEASURED-PENDING hypotheses — the live boot backfills both (see the next
    # test, which pins that they are MARKED as hypotheses in the file itself).
    assert senses.gpu_mem_util == 0.45
    assert senses.max_model_len == 262144


def test_orin_builtin_records_the_measured_budget_with_its_boot_order_caveat() -> None:
    # The two senses budget knobs ARE measurements now (live boot on a physical
    # Orin, 2026-08-04) — but they are senses-FIRST measurements, and a
    # gears-first boot gets materially less KV because gpu_mem_util is a fraction
    # of the whole device. A future reader copying either number without that
    # caveat is exactly the failure this pins against.
    text = _orin_toml_text()
    assert "MEASURED-PENDING" not in text, "backfilled — the pre-boot marker must be gone"
    assert "docs/evidence/2026-08-04-accept-senses-unsloth-orin.txt" in text
    assert "#171" in text  # the deviation/issue that re-measured the KV pool
    measured_block = text.split("MEASURED", 1)[1].split("[roles.embedder]", 1)[0]
    assert "gpu_mem_util = 0.45" in measured_block
    assert "max_model_len = 262144" in measured_block
    # the measured KV figures, so a silent edit of one number is caught
    assert "609,266" in measured_block
    assert "11.81 GiB" in measured_block
    # and the caveat that makes them safe to reuse
    assert "BOOT ORDER" in measured_block


def test_orin_builtin_marks_every_nvfp4_generate_lobe_infeasible() -> None:
    # Both stay declared infeasible, carrying no model and no knobs — but for
    # two DIFFERENT reasons, which lightning-on-orin t4 separated:
    #   * muse (nvidia/Gemma-4-31B-IT-NVFP4) quantizes ACTIVATIONS to FP4, which
    #     needs Blackwell tensor cores; sm_87 is Ampere. A hard architecture
    #     line, not a memory tradeoff.
    #   * worker (Lightning) is W4A16 weight-only per its own hf_quant_config
    #     and DID boot live on this board — it stays false only because no gear
    #     is declared for it here yet. See the two carve-out tests below.
    #
    # `cortex` is deliberately NOT in this list any more (qwen3-8-gguf-llamacpp
    # t5): the NVFP4 line is about the CHECKPOINT FORMAT, and this card serves
    # cortex from a weight-only GGUF through llama.cpp instead — see the
    # dedicated test below.
    orin = loader.load_builtin("orin")
    for role in ("muse", "worker"):
        rp = orin.role(role)
        assert rp.feasible is False, f"{role} must be infeasible on Ampere sm_87"
        assert rp == RoleProfile(feasible=False), f"{role} leaked a model/knob opinion"
    # The rationale is recorded next to the declaration, not only here.
    text = _orin_toml_text()
    assert "Blackwell" in text
    assert "W4A16" in text  # the contrasting weight-only scheme senses uses


# --- the W4A4 line is PER-CHECKPOINT, not per-board -------------------------
#
# lightning-on-orin t4. The Orin card used to blame ONE reason — "NVFP4
# quantizes activations to FP4, sm_87 is Ampere" — for all three NVFP4 generate
# lobes. That reason is checkpoint-specific, and it is FALSE for Lightning,
# whose own hf_quant_config.json declares MIXED_PRECISION with W4A16_NVFP4
# (weight-only) experts. These two tests pin the carve-out in BOTH directions:
# Lightning is carved out with its config cited, and the two checkpoints the
# W4A4 sentence still describes stay named and stay infeasible.

_W4A4_CHECKPOINTS = ("unsloth/Qwen3.8-27B-NVFP4", "nvidia/Gemma-4-31B-IT-NVFP4")
_LIGHTNING_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
_CARVE_OUT_MARKER = "Lightning carve-out"


def _split_on_carve_out(text: str) -> tuple[str, str]:
    """(the surviving W4A4 statement, the Lightning carve-out that follows)."""
    assert _CARVE_OUT_MARKER in text, "the carve-out must be a findable, named block"
    before, after = text.split(_CARVE_OUT_MARKER, 1)
    return before.split("NVFP4 architecture line", 1)[-1], after


def test_orin_w4a4_line_still_names_the_two_checkpoints_it_describes() -> None:
    """The W4A4 statement must survive intact for the checkpoints it is TRUE of.

    Carving Lightning out must not become "the Orin can serve NVFP4 after all".
    unsloth/Qwen3.8-27B-NVFP4 (as an NVFP4 export) and
    nvidia/Gemma-4-31B-IT-NVFP4 do quantize activations to FP4 and remain
    infeasible on Ampere sm_87 — declared, not merely narrated.
    """
    orin = loader.load_builtin("orin")

    # nvidia/Gemma-4-31B-IT-NVFP4 is the `muse` lobe: declared infeasible, and
    # carrying no model/knob opinion at all.
    assert orin.role("muse") == RoleProfile(feasible=False)
    # unsloth/Qwen3.8-27B-NVFP4 is the `cortex` NVFP4 export: the role is served
    # here, but ONLY off the weight-only GGUF — the NVFP4 export is never
    # declared on this card.
    assert orin.role("cortex").model != "unsloth/Qwen3.8-27B-NVFP4"

    for text in (_orin_toml_text(), machines_orin.__doc__ or ""):
        surviving, carve_out = _split_on_carve_out(text)
        assert "W4A4" in surviving or "activations" in surviving
        for checkpoint in _W4A4_CHECKPOINTS:
            assert checkpoint in surviving, f"{checkpoint} must stay named as W4A4-blocked"
        # ...and Lightning must NOT be one of them any more.
        assert _LIGHTNING_ID not in surviving
        assert _LIGHTNING_ID in carve_out


def test_orin_carves_lightning_out_citing_its_own_quant_config_and_the_live_run() -> None:
    """Both files must justify the carve-out from Lightning's OWN config.

    "we tried it and it worked" is not the correction — the correction is that
    the checkpoint never had FP4 activations to begin with, which its
    hf_quant_config.json states, and which a live sm_87 boot then confirmed.
    """
    for text in (_orin_toml_text(), machines_orin.__doc__ or ""):
        _, carve_out = _split_on_carve_out(text)
        assert "hf_quant_config.json" in carve_out  # the primary source
        assert "W4A16_NVFP4" in carve_out  # weight-only experts
        # Split from a composite `in_proj and out_proj` assertion: the two
        # projections are separate facts about the checkpoint, and a composite
        # reports only that "something was missing" rather than which.
        assert "in_proj" in carve_out  # the FP8 half, input projection
        assert "out_proj" in carve_out  # the FP8 half, output projection
        assert "FP8" in carve_out  # ...and the barrier half that IS real
        assert "docs/evidence/2026-08-25-spike-lightning-vllm-orin.txt" in carve_out
        assert "Marlin" in carve_out  # the kernel stack the live boot selected

    # The correction is to the REASON only — t4 flips no role. `worker` stays
    # declared infeasible here; hosting Lightning is t6/t8/t9's decision.
    assert loader.load_builtin("orin").role("worker") == RoleProfile(feasible=False)


def test_orin_builtin_serves_cortex_on_the_llama_cpp_engine() -> None:
    # The Ampere board CAN serve the cortex role — just not the NVFP4 export.
    # It declares the catalog's llama.cpp GGUF gear, and declares NONE of the
    # vLLM-only knobs, because `llama-server` has no flag to receive them (a
    # knob that reaches no flag is the dead declaration #92 forbids).
    from lobes.catalog import ENGINE_LLAMA_CPP
    from lobes.profiles.shape_render import role_engine

    cortex = loader.load_builtin("orin").role("cortex")
    assert cortex.feasible is True
    assert cortex.model == "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"
    assert role_engine(cortex) == ENGINE_LLAMA_CPP
    # The GGUF's own declared native ceiling, served in full at the t1 spike.
    assert cortex.max_model_len == 262144
    assert cortex.gpu_mem_util is None  # no utilization fraction on this engine
    assert cortex.quantization is None  # Q4_K_M lives inside the .gguf
    assert cortex.kv_cache_dtype is None
    assert cortex.attention_backend is None
    assert cortex.enforce_eager is None
    assert cortex.hf_overrides is None


def test_orin_builtin_records_the_co_residency_exclusion_next_to_the_declaration() -> None:
    # cortex (~33 GiB measured) and senses (0.45 x 61.3 = ~27.6 GiB) do not both
    # fit in 61.3 GiB with ZERO swap, so this card needs a SHAPE to pick one.
    # Both stay feasible (feasibility is "can the board serve it at all?"), which
    # makes the exclusion invisible in the data — so the file has to say it.
    orin = loader.load_builtin("orin")
    assert orin.role("cortex").feasible is True
    assert orin.role("senses").feasible is True
    text = _orin_toml_text()
    assert "CANNOT CO-RESIDE" in text
    assert "orin-cortex" in text  # the shape that resolves it
    assert "orin-lobe" in text  # ...and the other one


def test_orin_builtin_pooling_gears_are_single_sourced_from_machines_registry() -> None:
    # Same contract as thor's: the carried-over Jetson pooling divergences
    # (embedder/reranker TRITON_ATTN, reranker enforce_eager) are declared once
    # in lobes/machines/orin.py and overlaid at load time, never re-typed in the
    # TOML — so the two can never drift apart.
    orin = loader.load_builtin("orin")
    strategy = machines.get("orin")
    assert strategy is not None
    overlaid = strategy.role_knobs()
    assert set(overlaid) == {"embedder", "reranker"}
    for role, knobs in overlaid.items():
        role_profile = orin.role(role)
        for knob_name, knob in knobs.items():
            assert getattr(role_profile, knob_name) == knob.value
    # ...and the literals really are absent from the file (the "don't re-type
    # it" half of the contract, which the value assertions above cannot see).
    pooling_text = _orin_toml_text().split("[roles.embedder]", 1)[1]
    assert "attention_backend =" not in pooling_text
    assert "enforce_eager =" not in pooling_text

    # The pooling gears themselves stay the fleet-standard 0.6B pair.
    assert orin.role("embedder").model == "Qwen/Qwen3-Embedding-0.6B"
    assert orin.role("reranker").model == "Qwen/Qwen3-Reranker-0.6B"
    assert orin.role("embedder").gpu_mem_util == 0.06
    assert orin.role("reranker").gpu_mem_util == 0.06


def test_orin_builtin_round_trips() -> None:
    orin = loader.load_builtin("orin")
    again = Profile.from_dict("orin", orin.to_dict())
    assert again == orin


def test_orin_builtin_leaves_spark_and_thor_untouched() -> None:
    # The plan boundary: adding a card must not move another card's rendering.
    # (tests/goldens/ pins this byte-for-byte; this is the cheap in-suite echo.)
    spark = loader.load_builtin("spark")
    thor = loader.load_builtin("thor")
    orin = loader.load_builtin("orin")
    assert spark.role("senses").model == "coolthor/gemma-4-12B-it-NVFP4A16"
    assert thor.role("senses").model == "coolthor/gemma-4-12B-it-NVFP4A16"
    assert orin.role("senses").model != spark.role("senses").model
    assert spark.role("cortex").feasible is True
    assert thor.role("cortex").feasible is True


# --- loader: resolution + operator overrides --------------------------------


def test_builtin_names_lists_spark_and_thor() -> None:
    names = loader.builtin_names()
    assert "spark" in names
    assert "thor" in names


def test_builtin_names_includes_orin() -> None:
    assert "orin" in loader.builtin_names()


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
    assert builtin_spark.role("cortex").model == "unsloth/Qwen3.8-27B-NVFP4"


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


# --- Qodo #176: host_env values are written to .env verbatim ------------------


def test_host_env_rejects_a_newline_that_would_corrupt_the_env_file() -> None:
    """`.env` is line-oriented, so an embedded newline SPLITS the entry.

    It does not escape — the tail becomes a bogus key or a parse error, and the
    failure surfaces far from the profile that caused it. Reject it while the
    operator still has the profile in front of them.
    """
    for bad in ("100\nEVIL=1", "100\rEVIL=1", "100\x00"):
        with pytest.raises(ModelGearError) as excinfo:
            Profile.from_dict("orin", {"host_env": {"LOBES_IOWAIT_DEGRADED_THRESHOLD": bad}})
        assert "newline" in str(excinfo.value).lower() or "nul" in str(excinfo.value).lower()


def test_host_env_accepts_ordinary_single_line_values() -> None:
    p = Profile.from_dict("orin", {"host_env": {"LOBES_IOWAIT_DEGRADED_THRESHOLD": "100"}})
    assert p.host_env == {"LOBES_IOWAIT_DEGRADED_THRESHOLD": "100"}


def test_host_env_key_pattern_stays_ascii_not_unicode_word_chars() -> None:
    """SonarCloud S6353 suggests `\\w` for `[A-Za-z0-9_]`. It is NOT equivalent.

    Python's `\\w` is Unicode-aware by default, so swapping it in would widen the
    accepted key set to include non-ASCII names — which would then be written
    verbatim into `.env`. This pins the ASCII-only contract so the "concise
    character class" refactor cannot land silently.
    """
    for bad_key in ("CAFÉ_VAR", "Aπ", "ПЕРЕМЕННАЯ"):
        with pytest.raises(ModelGearError) as exc:
            Profile.from_dict("x", {"host_env": {bad_key: "1"}})
        assert exc.value.code == EXIT_USER_ERROR
        assert bad_key in exc.value.message
    # the ASCII form still works
    assert Profile.from_dict("x", {"host_env": {"LOBES_OK_1": "1"}}).host_env == {"LOBES_OK_1": "1"}


def test_a_third_engine_never_borrows_the_llama_cpp_lane() -> None:
    # Qodo #200-2 (HIGH, correctness). `LLAMA_CPP_ACTIVATION_ENV` is keyed by ROLE,
    # so an ungated `.get(role)` handed ANY non-vLLM engine the llama.cpp activation:
    # a cortex gear declaring engine="sglang" (added 0.60.0) silently rendered
    # PRIMARY_URL=http://llamacpp-primary:8000 + COMPOSE_PROFILES=llamacpp, pointing
    # the gateway at a `llama-server` lane that cannot load it. shape_render already
    # gated on the engine; render.py did not, so the two disagreed. Both must refuse.
    from lobes.cli._errors import ModelGearError
    from lobes.profiles.render import _engine_activation_env
    from lobes.profiles.shape_render import role_service

    sglang_cortex = RoleProfile(feasible=True, model="RadixArk/Qwen3.8-27B-NVFP4")

    for call, surface in (
        (lambda: _engine_activation_env("cortex", sglang_cortex), "render"),
        (lambda: role_service("cortex", sglang_cortex), "shape_render"),
    ):
        with pytest.raises(ModelGearError) as excinfo:
            call()
        message = str(excinfo.value)
        assert "sglang" in message, f"{surface} did not name the engine it refused"
        assert "llamacpp-primary" not in message
