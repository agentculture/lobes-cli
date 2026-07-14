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

from lobes.profiles.loader import builtin_names, resolve_profile
from lobes.profiles.render import ROLE_ENV_PREFIX, profile_env
from lobes.profiles.schema import ROLES, Profile, RoleProfile
from lobes.profiles.shape_render import (
    AUDIO_OVERLAY_FILE,
    FLEET_COMPOSE_FILE,
    GATEWAY_SERVICE,
    OPT_IN_ACTIVATION_ENV,
    REALTIME_SERVICE,
    ROLE_SERVICE,
    compose_profile,
    render_shape,
    shape_compose_files,
    shape_env,
    shape_services,
)
from lobes.profiles.shapes import (
    AUDIO_ROLES,
    COLLEAGUE_ROLES,
    OPT_IN_ROLES,
    Shape,
    builtin_shape_names,
    resolve_shape,
)
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

    "Everything" means the six first-class Colleague roles
    (:data:`COLLEAGUE_ROLES`) -- NOT the broader
    :data:`~lobes.profiles.shapes.SHAPE_ROLES`, which also admits the opt-in
    `minor` gear (t2, issue #112) that machine-as-brain deliberately never
    hosts.
    """
    shape = resolve_shape(_IDENTITY_SHAPE)
    assert set(shape.hosts) == set(COLLEAGUE_ROLES)
    assert "minor" not in shape.hosts
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
    """No opt-in gear hosted -> no activation keys (machine-as-brain stays byte-identical)."""
    for shape_name in ("machine-as-brain", "spark-lobe", "thor-lobe"):
        shape = resolve_shape(shape_name)
        for card in builtin_names():
            env = shape_env(shape, resolve_profile(card))
            for key in ("COMPOSE_PROFILES", "MINOR_BASE_URL", "MINOR_SERVED_NAME"):
                assert key not in env, f"{shape_name}/{card} leaked {key}"


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
    """Across every shape x card, a dropped core role emits ONLY <PREFIX>_FEASIBLE=false."""
    for shape_name in builtin_shape_names():
        shape = resolve_shape(shape_name)
        for card in builtin_names():
            env = shape_env(shape, resolve_profile(card))
            for role in ROLES:
                if shape.hosts_role(role):
                    continue
                prefix = ROLE_ENV_PREFIX[role]
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


def test_audio_overlay_included_iff_audio_role_hosted() -> None:
    """The audio overlay compose file is present exactly when the shape hosts stt/tts."""
    for shape_name in builtin_shape_names():
        shape = resolve_shape(shape_name)
        files = shape_compose_files(shape)
        assert files[0] == FLEET_COMPOSE_FILE
        hosts_audio = any(shape.hosts_role(r) for r in AUDIO_ROLES)
        assert (AUDIO_OVERLAY_FILE in files) == hosts_audio


def test_gateway_always_serves_and_realtime_rides_the_overlay() -> None:
    """The gateway fronts every shape; the realtime bridge is up iff the overlay is."""
    for shape_name in builtin_shape_names():
        shape = resolve_shape(shape_name)
        services = shape_services(shape, resolve_profile("spark"))
        assert GATEWAY_SERVICE in services
        hosts_audio = any(shape.hosts_role(r) for r in AUDIO_ROLES)
        assert (REALTIME_SERVICE in services) == hosts_audio


def test_services_cover_exactly_the_hosted_feasible_roles() -> None:
    """Every hosted+feasible role has its compose service; dropped/infeasible roles do not."""
    for shape_name in builtin_shape_names():
        shape = resolve_shape(shape_name)
        for card in builtin_names():
            profile = resolve_profile(card)
            services = set(shape_services(shape, profile))
            for role in ROLES:
                service = ROLE_SERVICE[role]
                should_run = shape.hosts_role(role) and profile.role(role).feasible
                assert (
                    service in services
                ) == should_run, f"{role} service {service!r} presence wrong on {shape_name}/{card}"
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
        + (fleet_dir / AUDIO_OVERLAY_FILE).read_text(encoding="utf-8")
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
            assert rendered.compose_files
            assert rendered.services
