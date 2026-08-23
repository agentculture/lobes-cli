"""The co-residency guard: ``lobes init`` refuses a DEFAULT shape that over-hosts.

``feasible`` is a per-ROLE question — "can this board serve this role at all?"
— and both members of a co-residency clash answer it honestly with yes. The
Jetson AGX Orin can serve ``cortex`` (through the llama.cpp GGUF lane) and it
can serve ``senses``; it just cannot serve BOTH at once, because ~33 GiB +
~27.6 GiB does not fit in 61.3 GiB with zero swap. Nothing in
:class:`~lobes.profiles.schema.RoleProfile` can say that, since it is a
statement about a PAIR.

Before this guard the card's own banner said it in prose and the default path
ignored it: ``machine-as-brain`` (what a bare ``lobes init`` resolves) hosts
every feasible role, so ``lobes init --apply`` on an Orin rendered a deployment
expected to OOM at boot — a regression on the decision-free path.

The fix is a declaration, not a card-name branch. A profile may carry
``[[exclusive_roles]]`` groups (:class:`~lobes.profiles.schema.ExclusiveRoles`);
:func:`~lobes.profiles.shape_render.overcommitted_groups` reports the groups a
(shape, card) pair would actually run more than one member of; and ``init``
turns that into a :class:`~lobes.cli._errors.ModelGearError` naming the shapes
the card itself declares as the way out.

Two properties this module pins hardest:

* **Every other card is untouched.** No built-in profile but ``orin`` declares
  a group, so spark/thor/base render exactly as before — asserted here against
  the goldens' own source of truth as well as end-to-end through ``init``.
* **The declaration cannot drift into a lie.** Each shape a group names must
  exist AND must actually resolve the group.

Card detection is injected (``monkeypatch.setattr(_detect, "detect_card", …)``
— this repo's offline-probe idiom), so nothing here touches real hardware.
"""

from __future__ import annotations

import pytest

from lobes.cli import main
from lobes.cli._errors import EXIT_USER_ERROR
from lobes.profiles.loader import builtin_names, resolve_profile
from lobes.profiles.schema import ROLES, ExclusiveRoles, Profile
from lobes.profiles.shape_render import overcommitted_groups, shape_env
from lobes.profiles.shapes import builtin_shape_names, resolve_shape
from lobes.runtime import _compose, _detect

_DEFAULT_SHAPE = "machine-as-brain"
# The one built-in card that declares a co-residency limit today. Named as a
# fixture value, never matched on inside the implementation — the guard reads
# the DECLARATION, so a second card declaring one gets the behaviour for free.
_DECLARING_CARD = "orin"


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


def _patch_detect(monkeypatch, card: str) -> None:
    monkeypatch.setattr(_detect, "detect_card", lambda: _fake_card(card))


def _declaring_cards() -> list[str]:
    return [name for name in builtin_names() if resolve_profile(name).exclusive_roles]


# --- the declaration itself --------------------------------------------------


def test_exactly_one_builtin_card_declares_a_co_residency_limit() -> None:
    """Guards the "every other card is unaffected" claim at its source."""
    assert _declaring_cards() == [_DECLARING_CARD]


def test_the_declared_group_is_the_measured_cortex_senses_pair() -> None:
    groups = resolve_profile(_DECLARING_CARD).exclusive_roles
    assert len(groups) == 1
    group = groups[0]
    assert set(group.roles) == {"cortex", "senses"}
    assert set(group.roles) <= set(ROLES)
    # The refusal quotes the reason back at the operator; an empty one would
    # make the error unverifiable.
    assert "61.3" in group.reason


def test_the_machine_registry_overlay_does_not_drop_the_declaration() -> None:
    """A real bug, caught once: ``orin`` is a MACHINE-DERIVED built-in.

    ``lobes.profiles.loader._apply_machine_registry`` overlays the chip-strategy
    knobs by CONSTRUCTING a fresh :class:`Profile` rather than
    ``dataclasses.replace``-ing one, so any non-role field it forgets to name is
    silently dropped — for exactly the two machine-derived cards (``thor``,
    ``orin``) and nowhere else. ``orin`` is both the card that declares a group
    and one of the two that goes through that overlay, so the guard would have
    been dead on the only card that needs it.
    """
    from lobes.profiles.loader import load_builtin

    assert load_builtin(_DECLARING_CARD).exclusive_roles  # post-overlay
    assert resolve_profile(_DECLARING_CARD).exclusive_roles  # and post-resolution


def test_every_declared_resolving_shape_exists_and_actually_resolves_the_group() -> None:
    """The declaration cannot drift into a lie.

    ``shapes`` is declared data (deriving it would surface every shape on every
    card), so it is proven here rather than trusted: each named shape must be a
    real built-in AND must host at most one member of the group it claims to
    resolve.
    """
    for card in _declaring_cards():
        profile = resolve_profile(card)
        for group in profile.exclusive_roles:
            assert group.shapes, f"{card}: a group naming no resolving shape is a dead end"
            for name in group.shapes:
                assert name in builtin_shape_names(), f"{card}: unknown shape {name!r}"
                shape = resolve_shape(name)
                hosted = [role for role in group.roles if shape.hosts_role(role)]
                assert len(hosted) <= 1, f"{name} hosts {hosted} — it does not resolve the group"
                assert not overcommitted_groups(shape, profile)


# --- the predicate -----------------------------------------------------------


def test_default_shape_over_hosts_the_declaring_card() -> None:
    profile = resolve_profile(_DECLARING_CARD)
    over = overcommitted_groups(resolve_shape(_DEFAULT_SHAPE), profile)
    assert [tuple(g.roles) for g in over] == [("cortex", "senses")]


@pytest.mark.parametrize("card", [name for name in builtin_names() if name != _DECLARING_CARD])
@pytest.mark.parametrize("shape", builtin_shape_names())
def test_no_other_card_over_hosts_under_any_shape(card: str, shape: str) -> None:
    assert overcommitted_groups(resolve_shape(shape), resolve_profile(card)) == ()


def test_an_infeasible_member_is_not_a_clash() -> None:
    """The predicate reads the COMPOSED pair, so a card-vetoed role never counts.

    Two roles cannot fight over memory when one of them is not served at all —
    otherwise a card that declared a group and then turned a member off would
    refuse its own default shape forever.
    """
    profile = resolve_profile(_DECLARING_CARD)
    without_senses = Profile(
        name=profile.name,
        summary=profile.summary,
        roles={
            role: (rp if role != "senses" else type(rp)(feasible=False))
            for role, rp in profile.roles.items()
        },
        host_env=profile.host_env,
        gpu_access=profile.gpu_access,
        exclusive_roles=profile.exclusive_roles,
    )
    assert overcommitted_groups(resolve_shape(_DEFAULT_SHAPE), without_senses) == ()


def test_a_card_declaring_nothing_is_never_flagged() -> None:
    bare = Profile(name="custom", roles=dict(resolve_profile(_DECLARING_CARD).roles))
    assert bare.exclusive_roles == ()
    assert overcommitted_groups(resolve_shape(_DEFAULT_SHAPE), bare) == ()


# --- `lobes init`: the refusal ----------------------------------------------


def test_bare_init_refuses_on_the_declaring_card(tmp_path, monkeypatch, capsys) -> None:
    _patch_detect(monkeypatch, _DECLARING_CARD)
    target = tmp_path / "deploy"
    rc = main(["init", str(target), "--apply"])
    assert rc == EXIT_USER_ERROR
    err = capsys.readouterr().err
    assert "cortex" in err and "senses" in err
    # Names the concrete way out, both alternatives, from the card's own data.
    assert "orin-cortex" in err
    assert "orin-lobe" in err
    assert not target.exists(), "a refusal must not scaffold anything"


def test_bare_init_dry_run_refuses_too(tmp_path, monkeypatch, capsys) -> None:
    """A plan that quietly describes a deployment that cannot boot is the same
    bug one step earlier."""
    _patch_detect(monkeypatch, _DECLARING_CARD)
    target = tmp_path / "deploy"
    assert main(["init", str(target)]) == EXIT_USER_ERROR
    assert "orin-cortex" in capsys.readouterr().err
    assert not target.exists()


def test_a_forced_profile_refuses_the_same_way(tmp_path, monkeypatch, capsys) -> None:
    """The guard follows the RESOLVED profile, not the detected card — forcing
    ``--profile orin`` onto another box renders the same over-committed .env."""
    _patch_detect(monkeypatch, "spark")
    target = tmp_path / "deploy"
    assert main(["init", str(target), "--profile", _DECLARING_CARD, "--apply"]) == EXIT_USER_ERROR
    assert "orin-cortex" in capsys.readouterr().err


def test_the_refusal_quotes_the_measured_reason(tmp_path, monkeypatch, capsys) -> None:
    _patch_detect(monkeypatch, _DECLARING_CARD)
    assert main(["init", str(tmp_path / "d"), "--apply"]) == EXIT_USER_ERROR
    assert "61.3" in capsys.readouterr().err


# --- `lobes init`: what still works -----------------------------------------


@pytest.mark.parametrize("shape", ["orin-cortex", "orin-lobe"])
def test_a_resolving_shape_scaffolds_normally(tmp_path, monkeypatch, capsys, shape) -> None:
    _patch_detect(monkeypatch, _DECLARING_CARD)
    target = tmp_path / shape
    assert main(["init", str(target), "--shape", shape, "--apply"]) == 0
    capsys.readouterr()
    assert (target / _compose.ENV_FILE).exists()


def test_explicit_machine_as_brain_warns_but_proceeds(tmp_path, monkeypatch, capsys) -> None:
    """An operator who TYPES the shape has made the call knowingly.

    This mirrors ``--profile``'s own documented precedent ("forcing a profile
    onto a card it was not validated for … warns, but proceeds") rather than
    inventing a second override flag next to ``init``'s existing ``--force``.
    The one thing that must never be forceable is the DEFAULT path, and it
    isn't: the warning branch is reachable only by naming the shape.
    """
    _patch_detect(monkeypatch, _DECLARING_CARD)
    target = tmp_path / "deploy"
    assert main(["init", str(target), "--shape", _DEFAULT_SHAPE, "--apply"]) == 0
    captured = capsys.readouterr()
    assert "warning" in captured.err
    assert "cortex" in captured.err and "senses" in captured.err
    assert (target / _compose.ENV_FILE).exists()


def test_explicitly_named_default_renders_what_it_always_did(tmp_path, monkeypatch, capsys) -> None:
    """Proceeding means proceeding: the guard changes no rendered value."""
    _patch_detect(monkeypatch, _DECLARING_CARD)
    target = tmp_path / "deploy"
    assert main(["init", str(target), "--shape", _DEFAULT_SHAPE, "--apply"]) == 0
    capsys.readouterr()
    from lobes.runtime import _env

    env = _env.read_env_file(target / _compose.ENV_FILE)
    expected = shape_env(resolve_shape(_DEFAULT_SHAPE), resolve_profile(_DECLARING_CARD))
    for key, value in expected.items():
        assert key in env, f"{key} missing from the rendered .env"


# --- every other card is completely unaffected -------------------------------


@pytest.mark.parametrize("card", ["spark", "thor", "base"])
def test_bare_init_still_succeeds_on_every_other_card(tmp_path, monkeypatch, capsys, card) -> None:
    _patch_detect(monkeypatch, card)
    target = tmp_path / card
    assert main(["init", str(target), "--apply"]) == 0
    err = capsys.readouterr().err
    assert "cannot co-reside" not in err
    assert "mutually exclusive" not in err


@pytest.mark.parametrize("card", ["spark", "thor", "base"])
def test_bare_init_matches_explicit_default_shape_byte_for_byte(
    tmp_path, monkeypatch, capsys, card
) -> None:
    """The identity invariant the guard must not disturb on an unaffected card."""
    _patch_detect(monkeypatch, card)
    bare, explicit = tmp_path / "bare", tmp_path / "explicit"
    assert main(["init", str(bare), "--apply"]) == 0
    assert main(["init", str(explicit), "--shape", _DEFAULT_SHAPE, "--apply"]) == 0
    capsys.readouterr()
    assert (bare / _compose.ENV_FILE).read_text() == (explicit / _compose.ENV_FILE).read_text()


# --- schema validation: a declaration is never silently dropped --------------


def test_exclusive_roles_round_trips_through_to_dict() -> None:
    profile = resolve_profile(_DECLARING_CARD)
    again = Profile.from_dict(_DECLARING_CARD, profile.to_dict())
    assert again.exclusive_roles == profile.exclusive_roles
    assert again.to_dict() == profile.to_dict()


@pytest.mark.parametrize(
    "bad",
    [
        {"exclusive_roles": "cortex,senses"},
        {"exclusive_roles": [["cortex", "senses"]]},
        {"exclusive_roles": [{"roles": ["cortex"], "shapes": ["x"]}]},
        {"exclusive_roles": [{"roles": ["cortex", "cortex"], "shapes": ["x"]}]},
        {"exclusive_roles": [{"roles": ["cortex", "not_a_role"], "shapes": ["x"]}]},
        {"exclusive_roles": [{"roles": ["cortex", "senses"]}]},
        {"exclusive_roles": [{"roles": ["cortex", "senses"], "shapes": ["x"], "why": "no"}]},
        {"exclusive_roles": [{"roles": ["cortex", "senses"], "shapes": ["x"], "reason": 7}]},
    ],
)
def test_a_malformed_declaration_is_a_load_error(bad) -> None:
    with pytest.raises(Exception) as exc:
        Profile.from_dict("bogus", bad)
    assert getattr(exc.value, "code", None) == EXIT_USER_ERROR


def test_a_well_formed_declaration_loads() -> None:
    group = Profile.from_dict(
        "bogus",
        {
            "exclusive_roles": [
                {"roles": ["cortex", "senses"], "shapes": ["orin-lobe"], "reason": "because"}
            ]
        },
    ).exclusive_roles
    assert group == (
        ExclusiveRoles(roles=("cortex", "senses"), shapes=("orin-lobe",), reason="because"),
    )
