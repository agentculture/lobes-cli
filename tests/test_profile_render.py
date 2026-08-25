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
from lobes.profiles.shape_render import render_shape
from lobes.profiles.shapes import resolve_shape

# --- role -> prefix table ----------------------------------------------------


def test_role_env_prefix_covers_all_six_roles() -> None:
    assert ROLE_ENV_PREFIX == {
        "cortex": "PRIMARY",
        "senses": "MULTIMODAL",
        "muse": "MUSE",
        "worker": "WORKER",
        "hand": "HAND",
        "embedder": "EMBED",
        "reranker": "RERANK",
    }


# --- built-in profiles: spot-check the real mapping -------------------------


def test_spark_profile_env_matches_compose_defaults() -> None:
    spark = resolve_profile("spark")
    env = profile_env(spark)
    assert env["PRIMARY_MODEL"] == "unsloth/Qwen3.8-27B-NVFP4"
    assert env["PRIMARY_SERVED_NAME"] == "unsloth/Qwen3.8-27B-NVFP4"
    assert env["PRIMARY_GPU_MEM_UTIL"] == "0.3"
    assert env["PRIMARY_MAX_MODEL_LEN"] == "131072"
    assert env["PRIMARY_QUANTIZATION"] == "compressed-tensors"
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


# --- t5: spark-lobe's 1M YaRN hypothesis renders the three new knobs -------


# The argv token the cortex lane must end up running, byte-for-byte identical
# to the one `docker inspect` read off the deployed container on 2026-08-25
# (docs/evidence/2026-08-24-spike-dspark-cortex-spark.txt, Section 14).
_DSPARK_ARGV = (
    "--speculative-config="
    '{"method":"dspark","model":"RadixArk/Qwen3.8-27B-DSpark",'
    '"revision":"85ef153be924f17ce4bf62726954eeaa4a73e854",'
    '"num_speculative_tokens":7}'
)


def test_spark_lobe_render_carries_the_dspark_262k_knobs() -> None:
    """The ADOPTED shape (d4, 2026-08-25): DSpark at the native 262144 window.

    Supersedes the 1M-YaRN assertions this test carried until 2026-08-25. The
    1M window was withdrawn because DSpark and 1M cannot both be served at
    gpu_mem_util 0.58 on the GB10 — vLLM refused the boot outright. See
    spark-lobe.toml's d4 block and docs/dspark-speculation.md.
    """
    spark = resolve_profile("spark")
    spark_lobe = resolve_shape("spark-lobe")
    env = render_shape(spark_lobe, spark).env
    assert env["PRIMARY_MODEL"] == "unsloth/Qwen3.8-27B-NVFP4"
    # The checkpoint's own native ceiling — no longer a YaRN-extended reach.
    assert env["PRIMARY_MAX_MODEL_LEN"] == "262144"
    # Dropped with the 1M window: at exactly 262144 nothing needs a
    # ceiling-bypass, so the shape must not arm one.
    assert "PRIMARY_ALLOW_LONG_MAX_MODEL_LEN" not in env
    import json

    # KEPT, deliberately: every DSpark arm was measured with this rope config
    # in force, so removing it would render a shape nothing has been measured
    # under. See the d4 block in spark-lobe.toml.
    hf_overrides = json.loads(env["PRIMARY_HF_OVERRIDES"])
    rope = hf_overrides["text_config"]["rope_parameters"]
    assert rope["rope_type"] == "yarn"
    assert rope["factor"] == 4.0
    assert rope["original_max_position_embeddings"] == 262144
    # Every other stock rope field is preserved byte-for-byte (verified
    # against the checkpoint's own config.json — see the spark-lobe.toml
    # comment).
    assert rope["mrope_interleaved"] is True
    assert rope["mrope_section"] == [11, 11, 10]
    assert rope["partial_rotary_factor"] == 0.25
    assert rope["rope_theta"] == 10000000


def test_spark_lobe_speculative_config_survives_both_quoting_layers() -> None:
    """The rendered .env value must reach vLLM as ONE intact argv token.

    The compose slot is UNQUOTED —
    ``${PRIMARY_SPECULATIVE_CONFIG-'--speculative-config={...}'}`` — because
    the DEFAULT carries its own single quotes. So a value substituted there
    crosses TWO parsers, and must survive both:

    1. compose's dotenv reader, which strips an outer quote pair and expands
       ``\"`` escapes inside a double-quoted value; then
    2. compose's shell-lexer, which splits the command string on whitespace
       and consumes any quotes it finds.

    A value quoted for only one layer degrades SILENTLY: the bare-single-quote
    and unquoted spellings both survive dotenv and are then stripped by the
    lexer into ``{method:dspark,...}`` — no error, an invalid config, and vLLM
    fails at boot far from the cause. Checked against the real template with
    ``docker compose config``; this test is that check's offline standing
    guard, so the pipeline below deliberately MODELS the two parsers rather
    than asserting the opaque byte string alone.
    """
    import shlex

    env = render_shape(resolve_shape("spark-lobe"), resolve_profile("spark")).env
    raw = env["PRIMARY_SPECULATIVE_CONFIG"]

    # Layer 1 — dotenv: a double-quoted value keeps its inner quotes.
    assert raw.startswith('"') and raw.endswith('"'), raw
    after_dotenv = raw[1:-1].replace('\\"', '"')
    # Layer 2 — shell-lexer: the surviving single quotes make it one token.
    tokens = shlex.split(after_dotenv)
    assert tokens == [_DSPARK_ARGV], tokens

    # And the JSON inside is still JSON (the failure mode is that it isn't).
    import json

    payload = json.loads(tokens[0].split("=", 1)[1])
    assert payload["method"] == "dspark"
    assert payload["model"] == "RadixArk/Qwen3.8-27B-DSpark"
    # PINNED: a draft model is executable weights, so an unpinned revision
    # would make the next pull a silent config change.
    assert payload["revision"] == "85ef153be924f17ce4bf62726954eeaa4a73e854"
    assert payload["num_speculative_tokens"] == 7


def test_only_spark_lobe_renders_the_cortex_lane_overrides() -> None:
    # Every OTHER card/shape stays untouched by t5's YaRN wiring and by d4's
    # DSpark adoption — no other built-in shape declares any of these knobs.
    for card_name in ("spark", "thor", "base"):
        card = resolve_profile(card_name)
        for shape_name in ("machine-as-brain", "thor-lobe", "thor-muse", "thor-worker"):
            shape = resolve_shape(shape_name)
            env = render_shape(shape, card).env
            assert "PRIMARY_HF_OVERRIDES" not in env
            assert "PRIMARY_ALLOW_LONG_MAX_MODEL_LEN" not in env
            assert "PRIMARY_SPECULATIVE_CONFIG" not in env


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


# --- host_env: the card-level (non-role) .env passthrough --------------------


def test_host_env_is_rendered_verbatim_alongside_the_role_keys() -> None:
    profile = Profile(
        name="quirky-card",
        roles={"senses": RoleProfile(gpu_mem_util=0.45)},
        host_env={"LOBES_IOWAIT_DEGRADED_THRESHOLD": "100"},
    )
    assert profile_env(profile) == {
        "MULTIMODAL_GPU_MEM_UTIL": "0.45",
        "LOBES_IOWAIT_DEGRADED_THRESHOLD": "100",
    }


def test_a_role_knob_always_wins_a_host_env_name_collision() -> None:
    # host_env is rendered FIRST precisely so it can never shadow a lane's own
    # budget: a card that (wrongly) spells a role key there loses to the role
    # table, rather than silently overriding a measured knob.
    profile = Profile(
        name="collision",
        roles={"senses": RoleProfile(gpu_mem_util=0.45)},
        host_env={"MULTIMODAL_GPU_MEM_UTIL": "0.99"},
    )
    assert profile_env(profile)["MULTIMODAL_GPU_MEM_UTIL"] == "0.45"


def test_a_profile_with_no_host_env_renders_exactly_what_it_always_did() -> None:
    # The "byte-identical for every other card" guarantee, as a unit fact:
    # spark/thor/base declare no host_env, so nothing new appears in their env.
    for card in ("base", "spark", "thor"):
        profile = resolve_profile(card)
        assert dict(profile.host_env) == {}
        assert not any(k.startswith("LOBES_") for k in profile_env(profile))
