"""Golden rendered artifacts per (shape, card) pair — brain-shapes t3.

Rendering a ``(Shape, Profile)`` pair into the concrete compose/.env a box
runs (:func:`lobes.profiles.shape_render.render_shape`) is a PURE function of
``(shape, profile, template)`` — no GPU probe, no host read, no subprocess —
so these goldens run identically on any dev box, a GPU-less CI runner
included. The composition is layered over the two landed axes:

* the #108 per-machine :class:`~lobes.profiles.schema.Profile` (how each role
  is TUNED on a card) via :func:`lobes.profiles.render.profile_env`, and
* the brain-shapes t1 :class:`~lobes.profiles.shapes.Shape` (which roles a box
  HOSTS at all).

Acceptance criteria this suite encodes (brain-shapes t3):

1. The whole-brain ``machine-as-brain`` shape renders BYTE-IDENTICALLY to the
   pre-change per-card rendering — asserted against the EXISTING profile
   golden path (``tests/goldens/<card>.env``), never a copied duplicate; the
   ``spark-lobe`` render carries no ``senses`` service and the ``thor-lobe``
   render carries no ``cortex`` service (a dropped role shows the #110
   flagged-off ``<PREFIX>_FEASIBLE=false`` marker and no model/knobs, and its
   compose service is absent).
2. A change to one shape's data leaves every OTHER (shape, card) golden
   byte-identical — structurally guaranteed because each golden is generated
   from exactly one ``Shape`` + one ``Profile`` (``shape_env_text`` below),
   and pinned by the byte-for-byte suite.

Regenerate every (shape, card) golden — plus the profile/template goldens — with
the SAME one deterministic command as the profile goldens::

    uv run python tests/goldens/regen.py

then diff before committing (a golden moving that you didn't intend to touch is
the signal this suite exists to catch). The additions here are purely additive:
``tests/test_profile_goldens.py`` and the flat ``tests/goldens/*.env`` goldens
are untouched.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from lobes.catalog import ENGINE_LLAMA_CPP, ENGINE_VLLM
from lobes.profiles.loader import builtin_names, resolve_profile
from lobes.profiles.render import (
    LLAMA_CPP_ACTIVATION_ENV,
    LLAMA_CPP_COMPOSE_PROFILE,
    ROLE_ENV_PREFIX,
    profile_env,
    role_engine,
)
from lobes.profiles.schema import ROLES, Profile, RoleProfile
from lobes.profiles.shape_render import (
    GATEWAY_SERVICE,
    LLAMA_CPP_ROLE_SERVICE,
    OPT_IN_ACTIVATION_ENV,
    REALTIME_SERVICE,
    ROLE_SERVICE,
    compose_profile,
    render_shape,
    role_service,
    shape_env,
    shape_services,
)
from lobes.profiles.shapes import (
    AUDIO_ROLES,
    DEFAULT_HOSTED_ROLES,
    OPT_IN_CORE_ROLES,
    OPT_IN_ROLES,
    Shape,
    builtin_shape_names,
    resolve_shape,
)
from lobes.runtime import _compose
from tests.goldens.regen import (
    FLEET_COMPOSE,
    shape_env_text,
    shape_golden_pairs,
    shape_golden_path,
)

_PROFILE_GOLDENS_DIR = Path(__file__).resolve().parent / "goldens"
_SHAPES_GOLDENS_DIR = _PROFILE_GOLDENS_DIR / "shapes"
_REGEN_CMD = "uv run python tests/goldens/regen.py"

# The identity shape: hosts every role, no overrides -> renders identically to
# the bare card profile, so it is validated against the EXISTING profile
# goldens (tests/goldens/<card>.env), never copied into a shapes/ golden.
_IDENTITY_SHAPE = "machine-as-brain"


# --- criterion 1: machine-as-brain is a no-op (byte-identical to profile golden)


@pytest.mark.parametrize("card", builtin_names())
def test_machine_as_brain_is_byte_identical_to_profile_golden(card: str) -> None:
    """The whole-brain shape renders exactly what the bare card profile already did.

    Asserts equality against the EXISTING golden output path
    (``tests/goldens/<card>.env``), not a duplicated copy — machine-as-brain
    hosts every role and carries no overrides, so composing it over any card
    must change nothing about that card's rendering.
    """
    shape = resolve_shape(_IDENTITY_SHAPE)
    profile = resolve_profile(card)
    rendered = render_shape(shape, profile).env_text()
    existing_golden = (_PROFILE_GOLDENS_DIR / f"{card}.env").read_text(encoding="utf-8")
    assert rendered == existing_golden, (
        f"machine-as-brain on card {card!r} must render byte-identically to the "
        f"pre-change tests/goldens/{card}.env — the whole-brain shape is a no-op. "
        f"If this is a deliberate change, regenerate with: {_REGEN_CMD}"
    )


@pytest.mark.parametrize("card", builtin_names())
def test_machine_as_brain_env_equals_profile_env(card: str) -> None:
    """The composed env dict itself is identical to profile_env(profile), not just the text."""
    profile = resolve_profile(card)
    shape = resolve_shape(_IDENTITY_SHAPE)
    assert render_shape(shape, profile).env == profile_env(profile)


def test_machine_as_brain_carries_no_overrides_and_hosts_everything() -> None:
    """Guards the invariant the no-op property rests on (matches the t1 shape data).

    "Everything" means the six DEFAULT-hosted Colleague roles
    (:data:`DEFAULT_HOSTED_ROLES`) -- NOT the broader
    :data:`~lobes.profiles.shapes.SHAPE_ROLES`, which also admits the opt-in
    `minor` gear (t2, issue #112) and the opt-in core `muse` lobe, both of
    which machine-as-brain deliberately never hosts.
    """
    shape = resolve_shape(_IDENTITY_SHAPE)
    assert set(shape.hosts) == set(DEFAULT_HOSTED_ROLES)
    assert "minor" not in shape.hosts
    assert "muse" not in shape.hosts
    assert dict(shape.overrides) == {}


# --- byte-for-byte per-(shape, card) goldens --------------------------------


@pytest.mark.parametrize("shape_name,card", shape_golden_pairs())
def test_shape_golden_byte_for_byte(shape_name: str, card: str) -> None:
    """render_shape(shape, card) renders exactly what's committed under goldens/shapes/."""
    path = shape_golden_path(shape_name, card)
    if not path.is_file():
        pytest.fail(f"missing golden {path} — generate it with: {_REGEN_CMD}")
    actual = shape_env_text(shape_name, card)
    expected = path.read_text(encoding="utf-8")
    assert actual == expected, (
        f"tests/goldens/shapes/{shape_name}__{card}.env drifted from "
        f"render_shape(resolve_shape({shape_name!r}), resolve_profile({card!r})).\n"
        f"If this is a deliberate change, regenerate with: {_REGEN_CMD}\n"
        "then diff the result — a change to ONE shape's data should not move "
        "ANOTHER (shape, card) golden in the same diff."
    )


def test_shape_golden_file_set_matches_expected() -> None:
    """Every non-identity (shape, card) pair has a golden, and there are no strays."""
    on_disk = (
        {p.name for p in _SHAPES_GOLDENS_DIR.glob("*.env")}
        if _SHAPES_GOLDENS_DIR.is_dir()
        else set()
    )
    expected = {f"{shape}__{card}.env" for shape, card in shape_golden_pairs()}
    assert on_disk == expected, (
        "goldens/shapes/ is out of sync with shape_golden_pairs(); regenerate "
        f"with: {_REGEN_CMD}"
    )


def test_identity_shape_has_no_shapes_golden() -> None:
    """machine-as-brain is validated against the profile goldens, so it owns no shapes/ file."""
    pairs = {shape for shape, _ in shape_golden_pairs()}
    assert _IDENTITY_SHAPE not in pairs
    if _SHAPES_GOLDENS_DIR.is_dir():
        stray = list(_SHAPES_GOLDENS_DIR.glob(f"{_IDENTITY_SHAPE}__*.env"))
        assert stray == [], f"machine-as-brain must not own a shapes/ golden copy: {stray}"


# --- criterion 1: dropped role -> no running service ------------------------


def test_spark_lobe_renders_no_senses_service() -> None:
    """spark-lobe drops the Gemma senses lobe: flagged off, no model, no service."""
    shape = resolve_shape("spark-lobe")
    profile = resolve_profile("spark")
    rendered = render_shape(shape, profile)
    prefix = ROLE_ENV_PREFIX["senses"]  # MULTIMODAL
    assert rendered.env.get(f"{prefix}_FEASIBLE") == "false"
    assert f"{prefix}_MODEL" not in rendered.env
    leaked = [k for k in rendered.env if k.startswith(f"{prefix}_") and k != f"{prefix}_FEASIBLE"]
    assert leaked == [], f"dropped senses leaked knob env: {leaked}"
    assert ROLE_SERVICE["senses"] not in rendered.services
    # The committed golden carries the same honesty (a file-content fact).
    golden = shape_golden_path("spark-lobe", "spark").read_text(encoding="utf-8")
    assert f"{prefix}_MODEL" not in golden
    assert f"{prefix}_FEASIBLE=false" in golden


def test_thor_lobe_renders_no_cortex_service() -> None:
    """thor-lobe drops the Qwen cortex primary: flagged off, no model, no service."""
    shape = resolve_shape("thor-lobe")
    profile = resolve_profile("thor")
    rendered = render_shape(shape, profile)
    prefix = ROLE_ENV_PREFIX["cortex"]  # PRIMARY
    assert rendered.env.get(f"{prefix}_FEASIBLE") == "false"
    assert f"{prefix}_MODEL" not in rendered.env
    leaked = [k for k in rendered.env if k.startswith(f"{prefix}_") and k != f"{prefix}_FEASIBLE"]
    assert leaked == [], f"dropped cortex leaked knob env: {leaked}"
    assert ROLE_SERVICE["cortex"] not in rendered.services
    golden = shape_golden_path("thor-lobe", "thor").read_text(encoding="utf-8")
    assert f"{prefix}_MODEL" not in golden
    assert f"{prefix}_FEASIBLE=false" in golden


def test_orin_small_renders_no_cortex_or_senses_service() -> None:
    """orin-small (t2, issue #112) drops BOTH heavy generate lobes.

    Rendered against `spark` -- a card where cortex AND senses are BOTH
    feasible -- to prove this is the SHAPE's drop decision, not a side
    effect of the card marking them infeasible (the same pattern
    test_spark_lobe_renders_no_senses_service / test_thor_lobe_renders_no_cortex_service
    use above).
    """
    shape = resolve_shape("orin-small")
    profile = resolve_profile("spark")
    rendered = render_shape(shape, profile)
    for role in ("cortex", "senses"):
        prefix = ROLE_ENV_PREFIX[role]
        assert rendered.env.get(f"{prefix}_FEASIBLE") == "false"
        assert f"{prefix}_MODEL" not in rendered.env
        leaked = [
            k for k in rendered.env if k.startswith(f"{prefix}_") and k != f"{prefix}_FEASIBLE"
        ]
        assert leaked == [], f"dropped {role} leaked knob env: {leaked}"
        assert ROLE_SERVICE[role] not in rendered.services
    golden = shape_golden_path("orin-small", "spark").read_text(encoding="utf-8")
    assert "PRIMARY_MODEL" not in golden
    assert "MULTIMODAL_MODEL" not in golden
    assert "PRIMARY_FEASIBLE=false" in golden
    assert "MULTIMODAL_FEASIBLE=false" in golden


def test_orin_small_hosts_minor_service_on_every_card() -> None:
    """orin-small's generate lane is the opt-in `minor` gear (vllm-minor), always."""
    shape = resolve_shape("orin-small")
    assert set(OPT_IN_ROLES) == {"minor"}
    for card in builtin_names():
        services = shape_services(shape, resolve_profile(card))
        assert ROLE_SERVICE["minor"] in services
        assert ROLE_SERVICE["cortex"] not in services
        assert ROLE_SERVICE["senses"] not in services
        assert ROLE_SERVICE["embedder"] in services
        assert ROLE_SERVICE["reranker"] in services


def test_hosted_opt_in_role_renders_its_activation_env_on_every_card() -> None:
    """Hosting `minor` must ACTIVATE it, not just list its service (PR #121 Qodo find).

    vllm-minor is gated behind the `minor` Docker Compose profile and the
    gateway wires the backend only when MINOR_BASE_URL is non-empty — so a
    shape hosting the opt-in gear must render COMPOSE_PROFILES plus the
    wiring pair, or `docker compose up` starts nothing and `model=minor`
    404s on the very shape whose generate lane it is.
    """
    shape = resolve_shape("orin-small")
    for card in builtin_names():
        env = shape_env(shape, resolve_profile(card))
        assert env.get("COMPOSE_PROFILES") == "minor", f"minor profile not activated on {card}"
        assert env.get("MINOR_BASE_URL") == "http://vllm-minor:8000"
        assert env.get("MINOR_SERVED_NAME") == "Qwen/Qwen3.5-4B"


def test_shapes_without_opt_in_roles_render_no_activation_env() -> None:
    """No opt-in gear hosted -> no activation keys (machine-as-brain stays byte-identical).

    The `orin` CARD is excluded from the COMPOSE_PROFILES half on purpose: it
    serves cortex on the llama.cpp engine, whose lane is compose-profile-gated
    too, so any shape HOSTING cortex on that card legitimately renders
    COMPOSE_PROFILES=llamacpp (see
    test_llama_cpp_cortex_renders_its_lane_activation). The `minor` keys must
    still be absent everywhere — those are the opt-in GEAR's, and no shape here
    hosts it.
    """
    for shape_name in ("machine-as-brain", "spark-lobe", "thor-lobe"):
        shape = resolve_shape(shape_name)
        for card in builtin_names():
            profile = resolve_profile(card)
            env = shape_env(shape, profile)
            for key in ("MINOR_BASE_URL", "MINOR_SERVED_NAME"):
                assert key not in env, f"{shape_name}/{card} leaked {key}"
            hosts_llama_cpp_role = any(
                shape.hosts_role(role)
                and compose_profile(shape, profile).role(role).feasible
                and role_engine(compose_profile(shape, profile).role(role)) != ENGINE_VLLM
                for role in ROLES
            )
            if hosts_llama_cpp_role:
                continue
            assert "COMPOSE_PROFILES" not in env, f"{shape_name}/{card} leaked COMPOSE_PROFILES"


# --- the ENGINE axis: a role whose model is a non-vLLM gear runs another lane -


def test_role_service_is_the_vllm_lane_for_every_vllm_gear() -> None:
    """The engine axis is inert for everything that predates it.

    Every role on every built-in card resolves to exactly ROLE_SERVICE unless
    its model is a non-vLLM catalog gear — which is what keeps every pre-engine
    (shape, card) rendering byte-identical.
    """
    for card in builtin_names():
        profile = resolve_profile(card)
        for role in ROLES:
            rp = profile.role(role)
            if role_engine(rp) != ENGINE_VLLM:
                continue
            assert role_service(role, rp) == ROLE_SERVICE[role]


def test_llama_cpp_cortex_runs_its_own_lane_not_the_vllm_one() -> None:
    """The orin card's cortex is a GGUF gear, so it must NOT resolve to vllm-primary.

    Serving a `.gguf` through `vllm serve` cannot work, so this is the whole
    point of the axis: the SERVICE moves with the engine while the ROLE (and
    therefore every caller-facing alias) does not.
    """
    profile = resolve_profile("orin")
    cortex = profile.role("cortex")
    assert role_engine(cortex) == ENGINE_LLAMA_CPP
    assert role_service("cortex", cortex) == LLAMA_CPP_ROLE_SERVICE["cortex"]
    assert role_service("cortex", cortex) != ROLE_SERVICE["cortex"]

    services = shape_services(resolve_shape("orin-cortex"), profile)
    assert LLAMA_CPP_ROLE_SERVICE["cortex"] in services
    assert ROLE_SERVICE["cortex"] not in services
    # senses is dropped by this shape; the cheap gears stay.
    assert ROLE_SERVICE["senses"] not in services
    assert ROLE_SERVICE["hand"] in services
    assert ROLE_SERVICE["embedder"] in services
    assert ROLE_SERVICE["reranker"] in services


def test_llama_cpp_cortex_renders_its_lane_activation() -> None:
    """Naming the GGUF model must also START and WIRE its lane, in one render.

    The llamacpp-primary service is parked behind the `llamacpp` compose
    profile and the gateway dials the origin it is given — so a render that
    moved the model without moving both would leave `model=cortex` pointed at a
    vLLM lane that is not running. Same failure mode OPT_IN_ACTIVATION_ENV
    exists to prevent, one axis over.
    """
    env = shape_env(resolve_shape("orin-cortex"), resolve_profile("orin"))
    assert env["PRIMARY_MODEL"] == "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"
    assert env["PRIMARY_SERVED_NAME"] == env["PRIMARY_MODEL"]
    assert env["PRIMARY_URL"] == LLAMA_CPP_ACTIVATION_ENV["cortex"]["PRIMARY_URL"]
    assert LLAMA_CPP_COMPOSE_PROFILE in env["COMPOSE_PROFILES"].split(",")
    # No gpu_mem_util reaches the lane — llama.cpp has no such flag.
    assert "PRIMARY_GPU_MEM_UTIL" not in env


def test_llama_cpp_constants_mirror_the_shipped_compose_template() -> None:
    """Same discipline as test_role_service_constants_exist_in_compose_templates
    and test_opt_in_activation_env_mirrors_the_compose_template: the constants
    are a MIRROR of the template, so read the template and prove they line up —
    the service key, its profile gate, and the gateway's overridable origin."""
    text = FLEET_COMPOSE.read_text(encoding="utf-8")
    service = LLAMA_CPP_ROLE_SERVICE["cortex"]
    assert re.search(rf"^  {re.escape(service)}:$", text, re.MULTILINE)
    block = text.split(f"  {service}:", 1)[1].split("\n  # A warm", 1)[0]
    assert re.search(
        rf"profiles:\s*\n\s*- {re.escape(LLAMA_CPP_COMPOSE_PROFILE)}", block
    ), "the llama.cpp lane lost its compose-profile gate"
    # No host-published port — reachable only on the compose net (t4 criterion 1).
    assert "ports:" not in block
    assert re.search(r"expose:\s*\n\s*- \"8000\"", block)
    origin = LLAMA_CPP_ACTIVATION_ENV["cortex"]["PRIMARY_URL"]
    assert f"{service}:8000" in origin
    assert "- PRIMARY_URL=${PRIMARY_URL:-http://vllm-primary:8000}" in text


def test_llama_cpp_lane_carries_no_vllm_flags() -> None:
    """t4 criterion 3, asserted against the shipped template rather than by eye.

    `llama-server` has none of these surfaces; a translated-looking flag here
    would be a lie about what the lane can do.
    """
    text = FLEET_COMPOSE.read_text(encoding="utf-8")
    block = text.split(f"  {LLAMA_CPP_ROLE_SERVICE['cortex']}:", 1)[1].split("\n  # A warm", 1)[0]
    command = block.split("    command:", 1)[1].split("    healthcheck:", 1)[0]
    for vllm_only in (
        "vllm",
        "--quantization",
        "--gpu-memory-utilization",
        "--max-model-len",
        "--tool-call-parser",
        "--reasoning-parser",
        "--speculative-config",
        "--served-model-name",
        "--trust-remote-code",
        "--enable-lora",
    ):
        assert vllm_only not in command, f"vLLM flag {vllm_only!r} leaked into the llama.cpp lane"
    # ...and it DOES carry the measured llama.cpp ones.
    for llama_flag in ("llama-server", "--n-gpu-layers", "--ctx-size", "--jinja", "--flash-attn"):
        assert llama_flag in command


def test_opt_in_activation_env_mirrors_the_compose_template() -> None:
    """OPT_IN_ACTIVATION_ENV mirrors the SHIPPED fleet template (kept honest here).

    Same design as test_role_service_constants_exist_in_compose_templates: the
    constant mirrors the template's own defaults, so read the template and
    prove the mirror still lines up — the served-name default, the profile
    gate, and the gateway's opt-in wiring key.
    """
    text = FLEET_COMPOSE.read_text(encoding="utf-8")
    served = OPT_IN_ACTIVATION_ENV["minor"]["MINOR_SERVED_NAME"]
    assert f"${{MINOR_SERVED_NAME:-{served}}}" in text
    assert OPT_IN_ACTIVATION_ENV["minor"]["MINOR_BASE_URL"] == "http://vllm-minor:8000"
    assert "- MINOR_BASE_URL=${MINOR_BASE_URL:-}" in text
    minor_block = text.split("  vllm-minor:", 1)[1].split("\n  vllm-", 1)[0]
    assert re.search(r"profiles:\s*\n\s*- minor", minor_block), "vllm-minor lost its profile gate"


def test_every_dropped_core_role_renders_only_the_feasible_marker() -> None:
    """Across every shape x card, a dropped DEFAULT core role emits ONLY
    <PREFIX>_FEASIBLE=false. A non-hosted OPT-IN core role (muse) instead
    passes the card's own declaration through: the base card's veto renders
    its marker, a silent card renders NOTHING (the gateway's OPT_IN_BACKENDS
    unwired-by-default rule carries the honesty) -- and never a model/knob
    leak either way."""
    for shape_name in builtin_shape_names():
        shape = resolve_shape(shape_name)
        for card in builtin_names():
            profile = resolve_profile(card)
            env = shape_env(shape, profile)
            for role in ROLES:
                if shape.hosts_role(role):
                    continue
                prefix = ROLE_ENV_PREFIX[role]
                if role in OPT_IN_CORE_ROLES:
                    expected = "false" if role in profile.roles else None
                    assert (
                        env.get(f"{prefix}_FEASIBLE") == expected
                    ), f"non-hosted opt-in {role} on {shape_name}/{card}: card passthrough broken"
                else:
                    assert (
                        env.get(f"{prefix}_FEASIBLE") == "false"
                    ), f"dropped {role} on {shape_name}/{card} lacks its flagged-off marker"
                stray = [k for k in env if k.startswith(f"{prefix}_") and k != f"{prefix}_FEASIBLE"]
                assert stray == [], f"dropped {role} on {shape_name}/{card} leaked {stray}"


# --- override composition (t2 fills these; the mechanism is proven here now) --


def test_shape_override_replaces_card_value_at_render() -> None:
    """A shape's per-role override wins over the card profile's value for that knob."""
    card = Profile(
        name="c",
        roles={
            "cortex": RoleProfile(feasible=True, model="m", gpu_mem_util=0.30, max_model_len=131072)
        },
    )
    shape = Shape(
        name="s",
        hosts=("cortex",),
        overrides={"cortex": RoleProfile(gpu_mem_util=0.60, max_model_len=262144)},
    )
    env = shape_env(shape, card)
    assert env["PRIMARY_GPU_MEM_UTIL"] == "0.6"
    assert env["PRIMARY_MAX_MODEL_LEN"] == "262144"
    # A knob the override is SILENT on flows through from the card unchanged.
    assert env["PRIMARY_MODEL"] == "m"


def test_absent_override_flows_card_value_through_unchanged() -> None:
    """Hosting a role with no override yields exactly the card profile's rendering for it."""
    card = Profile(
        name="c",
        roles={"embedder": RoleProfile(model="E", gpu_mem_util=0.06, max_model_len=8192)},
    )
    shape = Shape(name="s", hosts=("embedder",))
    env = shape_env(shape, card)
    assert env["EMBED_MODEL"] == "E"
    assert env["EMBED_GPU_MEM_UTIL"] == "0.06"
    assert env["EMBED_MAX_MODEL_LEN"] == "8192"


def test_override_does_not_flip_feasibility() -> None:
    """A hosted role's feasibility is the card's call — the shape override never sets it."""
    card = Profile(name="c", roles={"cortex": RoleProfile(feasible=False)})
    shape = Shape(name="s", hosts=("cortex",), overrides={"cortex": RoleProfile(gpu_mem_util=0.9)})
    composed = compose_profile(shape, card)
    # Card marks cortex infeasible; the override can't resurrect it.
    assert composed.role("cortex").feasible is False
    env = shape_env(shape, card)
    assert env.get("PRIMARY_FEASIBLE") == "false"
    assert "PRIMARY_GPU_MEM_UTIL" not in env  # infeasible role renders no knobs


# --- the compose side of "compose/.env" -------------------------------------


def test_gateway_always_serves_and_realtime_rides_the_overlay() -> None:
    """The gateway fronts every shape; the realtime bridge is up iff the overlay is."""
    for shape_name in builtin_shape_names():
        shape = resolve_shape(shape_name)
        services = shape_services(shape, resolve_profile("spark"))
        assert GATEWAY_SERVICE in services
        hosts_audio = any(shape.hosts_role(r) for r in AUDIO_ROLES)
        assert (REALTIME_SERVICE in services) == hosts_audio


def test_services_cover_exactly_the_hosted_feasible_roles() -> None:
    """Every hosted+feasible role has its compose service; dropped/infeasible roles do not.

    "Its" service is engine-aware since the llama.cpp lane landed: the expected
    service is ``role_service(role, composed_role)``, which is ROLE_SERVICE for
    every vLLM gear and the alternative lane otherwise. The vLLM service of a
    role hosted on another engine must be ABSENT — that is what stops both
    cortex lanes running at once.
    """
    for shape_name in builtin_shape_names():
        shape = resolve_shape(shape_name)
        for card in builtin_names():
            profile = resolve_profile(card)
            composed = compose_profile(shape, profile)
            services = set(shape_services(shape, profile))
            for role in ROLES:
                role_profile = composed.role(role)
                should_run = shape.hosts_role(role) and role_profile.feasible
                service = role_service(role, role_profile) if should_run else ROLE_SERVICE[role]
                assert (
                    service in services
                ) == should_run, f"{role} service {service!r} presence wrong on {shape_name}/{card}"
                if should_run and service != ROLE_SERVICE[role]:
                    assert (
                        ROLE_SERVICE[role] not in services
                    ), f"{role} runs BOTH engines' lanes on {shape_name}/{card}"
            for role in AUDIO_ROLES:
                assert (ROLE_SERVICE[role] in services) == shape.hosts_role(role)


def test_role_service_constants_exist_in_compose_templates() -> None:
    """The role->service map mirrors the SHIPPED compose files (kept honest here).

    Same design as render.ROLE_ENV_PREFIX: the constant is a mirror of the
    template, so a test reads the template and proves the mirror still lines up.
    """
    fleet_dir = FLEET_COMPOSE.parent
    combined = (
        FLEET_COMPOSE.read_text(encoding="utf-8")
        + "\n"
        + (fleet_dir / _compose.AUDIO_OVERLAY).read_text(encoding="utf-8")
    )
    service_keys = set(re.findall(r"^  ([a-z][a-z0-9-]*):\s*$", combined, re.MULTILINE))
    for role, service in ROLE_SERVICE.items():
        assert service in service_keys, f"{service!r} (role {role}) is not a compose service"
    assert GATEWAY_SERVICE in service_keys
    assert REALTIME_SERVICE in service_keys


# --- criterion 2 / purity: no host state, no cross-shape coupling ------------


def test_shape_rendering_is_pure_repeated_calls_are_identical() -> None:
    """render_shape twice yields byte-identical output — no hidden state."""
    for shape_name in builtin_shape_names():
        shape = resolve_shape(shape_name)
        for card in builtin_names():
            profile = resolve_profile(card)
            first_env = render_shape(shape, profile).env
            second_env = render_shape(shape, profile).env
            assert first_env == second_env
            first_text = shape_env_text(shape_name, card)
            second_text = shape_env_text(shape_name, card)
            assert first_text == second_text


def test_shape_render_depends_only_on_its_own_shape_and_card() -> None:
    """Rendering one shape never consults another — order-independent, cross-shape isolated.

    Structural backing for acceptance criterion 2: each golden is a pure
    function of exactly one (shape, card), so rendering a DIFFERENT shape in
    between cannot perturb it.
    """
    spark_lobe = resolve_shape("spark-lobe")
    thor_lobe = resolve_shape("thor-lobe")
    spark = resolve_profile("spark")
    first = render_shape(spark_lobe, spark).env_text()
    _ = render_shape(thor_lobe, spark).env_text()  # noqa: F841 — deliberate interleave
    _ = render_shape(resolve_shape(_IDENTITY_SHAPE), spark).env_text()
    again = render_shape(spark_lobe, spark).env_text()
    assert first == again


def test_shape_rendering_consults_no_host_state(tmp_path, monkeypatch) -> None:
    """render_shape completes with an empty HOME, a trimmed environment, and no subprocess.

    Proves the "no GPU or host state" acceptance criterion directly: shape x
    card rendering is a pure function of (shape, profile, template).
    """

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "shape rendering must not spawn a subprocess "
            f"(called with args={args!r}, kwargs={kwargs!r})"
        )

    monkeypatch.setattr(subprocess, "run", _boom)
    monkeypatch.setattr(subprocess, "Popen", _boom)
    monkeypatch.setattr(os, "system", _boom)

    empty_home = tmp_path / "empty-home"
    empty_home.mkdir()
    monkeypatch.setenv("HOME", str(empty_home))
    for key in list(os.environ):
        if key not in ("HOME", "PATH"):
            monkeypatch.delenv(key, raising=False)

    for shape_name in builtin_shape_names():
        shape = resolve_shape(shape_name)
        for card in builtin_names():
            rendered = render_shape(shape, resolve_profile(card))
            assert rendered.env, f"{shape_name}/{card} rendered no env"
            assert rendered.services


# ---------------------------------------------------------------------------
# Rendered-knob reachability (hand-lobe plan, live finding 2026-08-10)
# ---------------------------------------------------------------------------
# A card profile that DECLARES a knob renders `<PREFIX>_<KNOB>=value` into the
# deployment's .env. If the compose template never substitutes that variable,
# the declaration is DEAD: `lobes init` writes it, the operator reads it as
# configured, and the lane ignores it.
#
# This is not hypothetical. The `hand` lane shipped consuming HAND_MODEL,
# HAND_SERVED_NAME, HAND_MAX_MODEL_LEN and HAND_GPU_MEM_UTIL but NOT
# HAND_ATTENTION_BACKEND — which builtin/orin.toml declares as TRITON_ATTN. The
# Orin therefore rendered an attention-backend choice the engine never saw.
# Caught by a live compose boot, not by any test; this is that test.


# COMPOSE_PROFILES is the ONE rendered key that is legitimately never
# ${substituted} by the template: `docker compose` reads it itself, as its own
# documented way of activating profile-gated services, so it reaches the lane by
# a different road than every knob below. (It is rendered whenever a card serves
# a role on a compose-profile-gated lane — today, orin's llama.cpp cortex.)
_NOT_SUBSTITUTED_BY_TEMPLATE = frozenset({"COMPOSE_PROFILES"})


@pytest.mark.parametrize("card_name", sorted(builtin_names()))
def test_every_rendered_profile_knob_is_substituted_by_the_fleet_template(card_name):
    """Every KEY a card profile renders must appear as ${KEY...} in the template.

    Guards the dead-knob class: a profile declaration that reaches no compose
    flag is silently inert, which reads as configured and is not.
    """
    template = FLEET_COMPOSE.read_text(encoding="utf-8")
    rendered = profile_env(resolve_profile(card_name))
    dead = [
        key
        for key in rendered
        if key not in _NOT_SUBSTITUTED_BY_TEMPLATE
        and f"${{{key}" not in template
        and f"${{{key}}}" not in template
    ]
    assert not dead, (
        f"{card_name}: these profile-rendered keys are never substituted by "
        f"templates/fleet/docker-compose.yml, so the declaration is dead: {dead}"
    )
