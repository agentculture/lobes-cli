"""The ``orin-associate`` deployment shape (lightning-on-orin plan, t9).

A THIRD answer to the Orin's ``[cortex, senses, associate]`` co-residency
group (``builtin/orin.toml``'s ``[[exclusive_roles]]``): this shape hosts the
opt-in ``associate`` role (Nemotron 3.5 Lightning 30B-A3B, MEASURED live
2026-08-25) and drops BOTH heavy defaults (``cortex``, ``senses``) to a peer,
alongside ``hand`` + the two pooling gears and no audio overlay — structurally
``thor-muse``/``thor-worker`` for a THIRD opt-in core role and a different card.

Two things have to line up for the declaration to be more than paperwork:

1. **The shape's own render actually activates the lobe.** ``associate`` is
   an OPT_IN_CORE_ROLE, and — per approved deviation d1
   (``.devague/deliveries/lightning-on-orin.json``) — the ``orin`` CARD
   declares ``[roles.associate] feasible = false`` (the measured numbers kept
   there only as documentation). Composing a shape that HOSTS an opt-in core
   role the card marks infeasible is a scenario no prior shape exercised
   (``thor-muse``/``thor-worker``'s card, ``thor.toml``, leaves ``muse``/
   ``worker`` undeclared, i.e. implicitly feasible) — this module is the
   regression pin for the opt-in-core overlay path in shape_render,
   the fix that makes a hosted opt-in core role's own shape override win
   feasibility over a card that abstains.
2. **cortex/senses are dropped, honestly**, and **no audio is advertised**
   (sm_87 cannot serve the Parakeet STT image — the same fact orin-cortex and
   orin-lobe already record).

**HONESTY (#108).** ``orin-associate`` is DECLARED, NOT VALIDATED — no box has
booted this SHAPE and no acceptance transcript exists for it, even though the
LANE (Lightning on this board) and its BUDGET are both measured. Nothing here
asserts a validation claim beyond what the file itself states.
"""

from __future__ import annotations

from importlib.resources import files

import pytest

from lobes.cli import main
from lobes.cli._commands import init as init_cmd
from lobes.profiles.loader import builtin_names, resolve_profile
from lobes.profiles.render import ROLE_ENV_PREFIX
from lobes.profiles.shape_render import ROLE_SERVICE, render_shape, shape_env, shape_services
from lobes.profiles.shapes import AUDIO_ROLES, resolve_shape
from lobes.runtime import _compose, _detect, _env
from tests.goldens.regen import shape_golden_path

_SHAPE = "orin-associate"
_CARD = "orin"
_MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"


def _shape_toml(name: str) -> str:
    return (
        files("lobes.profiles.builtin_shapes").joinpath(f"{name}.toml").read_text(encoding="utf-8")
    )


def _fake_card(resolved: str) -> _detect.DetectedCard:
    return _detect.DetectedCard(
        resolved=resolved,
        device_name="NVIDIA Test",
        compute_capability="sm_87",
        total_memory_gb=61.3,
        hostname="test-host",
        device_tree_model=None,
        sources={},
    )


# --- 1. the shape's own data -------------------------------------------------


def test_hosts_associate_alone_and_drops_every_other_role() -> None:
    shape = resolve_shape(_SHAPE)
    assert set(shape.hosts) == {"associate"}
    assert not shape.hosts_role("cortex")
    assert not shape.hosts_role("senses")
    for role in AUDIO_ROLES:
        assert not shape.hosts_role(role), "sm_87 has no Parakeet image (see orin-lobe)"


def test_declares_the_full_associate_override_matching_the_card_documentation() -> None:
    """LOCKSTEP: the shape's numbers must equal builtin/orin.toml's own
    [roles.associate] documentation block — the two describe the same
    physical measurement."""
    rp = resolve_shape(_SHAPE).override("associate")
    assert rp.model == _MODEL_ID
    assert rp.gpu_mem_util == 0.80
    assert rp.max_model_len == 128000
    assert rp.quantization == "modelopt"
    assert rp.kv_cache_dtype == "bfloat16"

    card_text = files("lobes.profiles.builtin").joinpath("orin.toml").read_text(encoding="utf-8")
    assert "gpu_mem_util   0.56" in card_text
    assert "max_model_len  128000" in card_text
    assert "feasible = false" in card_text  # d1: the card abstains, docs only


def test_the_vendor_refusal_is_recorded_not_dropped() -> None:
    text = _shape_toml(_SHAPE)
    assert "0.70" in text
    assert "REFUSED" in text


def test_claims_no_shape_validation_though_the_lane_and_budget_are_measured() -> None:
    """#108: no box has booted this SHAPE — separate from the LANE, which is a
    confirmed GO, and the BUDGET, which is measured."""
    text = _shape_toml(_SHAPE)
    assert "DECLARED, NOT VALIDATED" in text
    assert "VALIDATED on" not in text
    assert "UNVALIDATED" in resolve_shape(_SHAPE).summary
    assert "MEASURED" in text


def test_no_audio_and_the_reason_is_inherited() -> None:
    text = _shape_toml(_SHAPE)
    assert "Parakeet" in text
    assert "sm_87" in text
    assert "CANNOT serve the audio overlay" in text


# --- 2. the render actually activates the lobe (card silent + shape override) ---


def test_associate_renders_feasible_with_its_full_declaration_despite_the_card_veto() -> None:
    """The regression this module exists to pin: the CARD says
    feasible=false for associate (d1), but THIS shape hosts it — so the
    composed role must be feasible, carry the model, and wire the compose
    lane, not collapse to a bare ASSOCIATE_FEASIBLE=false marker."""
    env = shape_env(resolve_shape(_SHAPE), resolve_profile(_CARD))
    assert "ASSOCIATE_FEASIBLE" not in env  # feasible=True renders no marker at all
    assert env["ASSOCIATE_MODEL"] == _MODEL_ID
    assert env["ASSOCIATE_SERVED_NAME"] == _MODEL_ID
    assert env["ASSOCIATE_GPU_MEM_UTIL"] == "0.8"
    assert env["ASSOCIATE_MAX_MODEL_LEN"] == "128000"
    assert env["ASSOCIATE_QUANTIZATION"] == "modelopt"
    assert env["ASSOCIATE_KV_CACHE_DTYPE"] == "bfloat16"
    assert env["ASSOCIATE_BASE_URL"] == "http://vllm-associate:8000"
    assert env["COMPOSE_PROFILES"] == "associate"


def test_dropped_cortex_and_senses_are_flagged_off_and_leak_no_knob() -> None:
    rendered = render_shape(resolve_shape(_SHAPE), resolve_profile(_CARD))
    for role in ("cortex", "senses"):
        prefix = ROLE_ENV_PREFIX[role]
        assert rendered.env.get(f"{prefix}_FEASIBLE") == "false"
        leaked = [
            k for k in rendered.env if k.startswith(f"{prefix}_") and k != f"{prefix}_FEASIBLE"
        ]
        assert leaked == [], f"dropped {role} leaked knob env: {leaked}"


def test_muse_and_worker_stay_untouched_by_this_shape() -> None:
    """Only associate is hosted here; muse/worker (the other two opt-in core
    roles) must pass through the card's own (silent) declaration unchanged."""
    env = shape_env(resolve_shape(_SHAPE), resolve_profile(_CARD))
    assert env.get("MUSE_FEASIBLE") == "false"
    assert env.get("WORKER_FEASIBLE") == "false"


def test_the_associate_compose_service_is_hosted_and_the_others_are_not() -> None:
    services = shape_services(resolve_shape(_SHAPE), resolve_profile(_CARD))
    assert ROLE_SERVICE["associate"] in services
    assert ROLE_SERVICE["cortex"] not in services
    assert ROLE_SERVICE["senses"] not in services
    # SOLO shape (operator decision 2026-08-26): hand, embedder and reranker are
    # dropped too, so their lanes must NOT be in the composed service set. Note
    # hand: it is the pressure-policy servable floor that every other built-in
    # shape hosts, so a box on this shape has no floor at all.
    for role in ("hand", "embedder", "reranker"):
        assert ROLE_SERVICE[role] not in services


def test_the_card_keeps_its_tegra_iowait_declaration_under_this_shape() -> None:
    env = shape_env(resolve_shape(_SHAPE), resolve_profile(_CARD))
    assert env["LOBES_IOWAIT_DEGRADED_THRESHOLD"] == "100"


@pytest.mark.parametrize("card", builtin_names())
def test_goldens_exist_for_every_card(card: str) -> None:
    assert shape_golden_path(_SHAPE, card).is_file()


# --- 3. exclusive_roles resolves the three-way group -------------------------


def test_orin_associate_resolves_the_co_residency_group_by_hosting_only_associate() -> None:
    from lobes.profiles.shape_render import overcommitted_groups

    shape = resolve_shape(_SHAPE)
    profile = resolve_profile(_CARD)
    assert not overcommitted_groups(shape, profile)
    hosted = [
        role for group in profile.exclusive_roles for role in group.roles if shape.hosts_role(role)
    ]
    assert hosted == ["associate"]


# --- 4. init end to end, and byte-for-byte restore ---------------------------


def test_end_to_end_init_render_activates_associate(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(_detect, "detect_card", lambda: _fake_card(_CARD))
    assert main(["init", str(tmp_path), "--profile", _CARD, "--shape", _SHAPE, "--apply"]) == 0

    env_text = (tmp_path / _compose.ENV_FILE).read_text(encoding="utf-8")
    assert f"ASSOCIATE_MODEL={_MODEL_ID}" in env_text
    assert "COMPOSE_PROFILES=associate" in env_text
    assert "ASSOCIATE_FEASIBLE=false" not in env_text
    assert "PRIMARY_FEASIBLE=false" in env_text
    assert "MULTIMODAL_FEASIBLE=false" in env_text

    shape_override = (tmp_path / _compose.SHAPE_OVERLAY).read_text(encoding="utf-8")
    assert "  vllm-primary:" in shape_override
    assert "  vllm-multimodal:" in shape_override
    assert "vllm-associate" not in shape_override  # hosted lane is never parked

    dropped = init_cmd._shape_dropped_services(resolve_shape(_SHAPE), resolve_profile(_CARD))
    assert dropped == ["vllm-embed", "vllm-hand", "vllm-multimodal", "vllm-primary", "vllm-rerank"]


def test_reapplying_orin_associate_after_a_different_shape_restores_byte_for_byte(
    tmp_path, monkeypatch
) -> None:
    """Acceptance 3: re-running with the previous shape restores the deployment
    byte-for-byte.

    The generated compose override (``docker-compose.shape.yml``) is a plain
    REWRITE every ``init`` — never merged — so it is compared in full. The
    ``.env`` file is merge-only BY DESIGN (``lobes init --help``'s own
    ``--force`` text: "NEVER overwrites .env ... existing lines are left
    untouched"), so switching to a DIFFERENT shape and back can leave a
    dropped role's stale knob values (e.g. ``orin-cortex``'s GGUF
    ``PRIMARY_MODEL``) sitting under this shape's own
    ``PRIMARY_FEASIBLE=false`` marker — a pre-existing property of every
    built-in shape's round trip, not something this task's scope touches. The
    restore this asserts is the one the merge model actually guarantees: every
    key THIS shape's own render has an opinion on comes back exactly.
    """
    monkeypatch.setattr(_detect, "detect_card", lambda: _fake_card(_CARD))
    target = tmp_path / "deploy"
    assert main(["init", "--profile", _CARD, "--shape", _SHAPE, str(target), "--apply"]) == 0
    first_shape_override = (target / _compose.SHAPE_OVERLAY).read_text()
    rendered_keys = set(shape_env(resolve_shape(_SHAPE), resolve_profile(_CARD)))
    first_rendered = {
        k: v
        for k, v in _env.read_env_file(target / _compose.ENV_FILE).items()
        if k in rendered_keys
    }

    # Switch away to orin-cortex (a shape that resolves the same exclusive_roles
    # group a different way)...
    assert (
        main(
            [
                "init",
                "--profile",
                _CARD,
                "--shape",
                "orin-cortex",
                str(target),
                "--apply",
                "--force",
            ]
        )
        == 0
    )
    mid_rendered = {
        k: v
        for k, v in _env.read_env_file(target / _compose.ENV_FILE).items()
        if k in rendered_keys
    }
    assert mid_rendered != first_rendered  # sanity: it actually changed

    # ...then switch back — the compose override restores byte-for-byte, and
    # every key THIS shape's own render sets comes back to its original value.
    assert (
        main(["init", "--profile", _CARD, "--shape", _SHAPE, str(target), "--apply", "--force"])
        == 0
    )
    assert (target / _compose.SHAPE_OVERLAY).read_text() == first_shape_override
    final_rendered = {
        k: v
        for k, v in _env.read_env_file(target / _compose.ENV_FILE).items()
        if k in rendered_keys
    }
    assert final_rendered == first_rendered
