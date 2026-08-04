"""csv-mode GPU access — the card-declared ``gpu_access`` knob (Orin variation t8).

This Jetson AGX Orin's NVIDIA container toolkit (1.19.1, ``mode = "auto"``)
resolves to the legacy **csv** mode, where the compose templates'
``deploy.resources.reservations.devices`` GPU request fails at container
create ("invoking the NVIDIA Container Runtime Hook directly … is not
supported") — ``docs/orin-profiles.md`` divergence 1. The live workaround was a
HAND EDIT of the scaffolded compose files, which a re-init reverts.

The fix is declarative and card-scoped: a :class:`~lobes.profiles.schema.Profile`
declares ``gpu_access = "runtime"`` and ``lobes init --apply`` GENERATES the
compose overrides that ``!reset`` each GPU service's ``deploy:`` stanza and set
``runtime: nvidia`` instead — on EVERY render, which is the property the hand
edit lacked. Docker Compose has no conditional-block syntax, so a second compose
file is the only thing that can express the alternative form; the override
mechanism itself is the one already proven by ``docker-compose.shape.yml``.

Every card that takes the default (``gpu_access = "devices"``) writes nothing at
all, which is what keeps a non-csv deployment byte-identical to before this knob
existed.

**UNVALIDATED live (#108):** these tests prove the render and the compose MERGE
(``docker compose config`` was run by hand on the real csv-mode board while this
landed); they do not prove a container CREATE, which only a live boot can.
"""

from __future__ import annotations

import json
import re

import pytest
import yaml

from lobes.cli import main
from lobes.cli._commands import init as init_cmd
from lobes.profiles.loader import builtin_names, load_builtin, resolve_profile
from lobes.profiles.schema import GPU_ACCESS_DEVICES, GPU_ACCESS_RUNTIME, Profile
from lobes.profiles.shape_render import compose_profile
from lobes.profiles.shapes import resolve_shape
from lobes.runtime import _compose, _detect

from .test_init_shape import _ComposeTagLoader, _fake_card

# The card whose live toolkit measured csv mode. Every other built-in stays on
# the default — pinned below, so promoting a second card is a deliberate edit.
CSV_MODE_CARD = "orin"


def _load_gpu_override(target, name: str) -> dict:
    """Parse a generated GPU override with the merge-tag-tolerant loader."""
    return yaml.load((target / name).read_text(encoding="utf-8"), Loader=_ComposeTagLoader)


def _template_gpu_services(rel_path: str) -> tuple[str, ...]:
    """The services declaring a ``deploy:`` (i.e. a GPU request) in a template.

    Re-derived from the shipped compose text so the constants in
    :mod:`lobes.runtime._compose` cannot drift from the templates they mirror —
    the same "verify the constant against the real file" discipline
    ``tests/test_shape_goldens.py`` applies to ``ROLE_SERVICE``.
    """
    from importlib.resources import files

    text = _compose._read_template(files("lobes.templates"), rel_path)
    services: list[str] = []
    current: str | None = None
    for line in text.splitlines():
        match = re.match(r"^  ([A-Za-z0-9_.-]+):\s*$", line)
        if match:
            current = match.group(1)
        elif line == "    deploy:" and current is not None:
            services.append(current)
    return tuple(services)


# --- the schema knob --------------------------------------------------------


def test_gpu_access_defaults_to_the_template_form() -> None:
    """A profile silent on gpu_access takes the deploy.resources form — today's
    behaviour, so every pre-existing profile renders unchanged."""
    assert Profile(name="x").gpu_access == GPU_ACCESS_DEVICES
    assert Profile.from_dict("x", {}).gpu_access == GPU_ACCESS_DEVICES


def test_gpu_access_round_trips_through_to_dict() -> None:
    profile = Profile.from_dict("x", {"gpu_access": GPU_ACCESS_RUNTIME})
    assert profile.to_dict()["gpu_access"] == GPU_ACCESS_RUNTIME
    assert Profile.from_dict("x", profile.to_dict()).gpu_access == GPU_ACCESS_RUNTIME


@pytest.mark.parametrize("bad", ["nvidia", "csv", "", True, 1, None])
def test_unknown_gpu_access_is_a_load_error(bad) -> None:
    """Loud, not silent — a typo must not quietly fall back to the default and
    leave a csv board asking for the GPU the way it refuses."""
    with pytest.raises(Exception) as exc:
        Profile.from_dict("x", {"gpu_access": bad})
    assert "gpu_access" in str(exc.value)


def test_only_the_measured_card_declares_csv_mode() -> None:
    """Provenance pin: `orin` is the ONE box whose toolkit was measured in csv
    mode (docs/orin-profiles.md). Promoting another card must be deliberate."""
    declared = {
        name for name in builtin_names() if load_builtin(name).gpu_access == GPU_ACCESS_RUNTIME
    }
    assert declared == {CSV_MODE_CARD}


def test_gpu_access_survives_the_machine_registry_overlay() -> None:
    """`orin` is a machine-derived built-in (loader._MACHINE_DERIVED_BUILTINS):
    the registry overlay rebuilds the Profile, and must carry the card's non-role
    declarations through — the same guarantee host_env already has."""
    assert load_builtin(CSV_MODE_CARD).gpu_access == GPU_ACCESS_RUNTIME


@pytest.mark.parametrize("shape", ["machine-as-brain", "orin-lobe", "orin-small"])
def test_gpu_access_passes_through_every_shape(shape: str) -> None:
    """Which compose syntax the runtime accepts is a fact about the BOARD, so no
    hosting decision may alter it — a shape-scoped fix would leave a bare
    `lobes init` on the same board broken (the host_env reasoning)."""
    card = resolve_profile(CSV_MODE_CARD)
    composed = compose_profile(resolve_shape(shape), card)
    assert composed.gpu_access == GPU_ACCESS_RUNTIME


# --- the generated overrides ------------------------------------------------


def test_default_cards_generate_nothing() -> None:
    """The byte-identity guarantee: a non-csv card renders no GPU override, so
    its deployment is exactly what it was before this knob existed."""
    for name in ("base", "spark", "thor"):
        assert init_cmd.render_gpu_overrides(resolve_profile(name)) is None


def test_service_constants_mirror_the_shipped_templates() -> None:
    """Drift-proofing: the constants naming which services get an override are
    re-derived from the real compose text. A new GPU gear cannot land in a
    template without landing here too — otherwise it would silently keep the
    deploy stanza and fail to create on a csv board."""
    assert _compose.GPU_SERVICES == _template_gpu_services("fleet/docker-compose.yml")
    assert _compose.GPU_SERVICES_AUDIO == _template_gpu_services("fleet/docker-compose.audio.yml")


def test_override_resets_deploy_and_asks_via_runtime() -> None:
    texts = init_cmd.render_gpu_overrides(resolve_profile(CSV_MODE_CARD))
    assert set(texts) == {_compose.GPU_OVERLAY, _compose.GPU_AUDIO_OVERLAY}
    expected = {
        _compose.GPU_OVERLAY: _compose.GPU_SERVICES,
        _compose.GPU_AUDIO_OVERLAY: _compose.GPU_SERVICES_AUDIO,
    }
    for name, services in expected.items():
        doc = yaml.load(texts[name], Loader=_ComposeTagLoader)
        assert tuple(doc["services"]) == services
        for service in services:
            block = doc["services"][service]
            # `!reset` CLEARS the attribute; `!override` would REPLACE it.
            assert block["deploy"] == {"__reset__": True}
            assert block["runtime"] == "nvidia"


def test_each_half_only_names_services_its_own_base_file_declares() -> None:
    """Why there are two files at all: compose rejects an override naming a
    service no file in the chain declares ("has neither an image nor a build
    context specified"), and the audio sidecars live in an OPT-IN overlay that
    `lobes up <non-audio-role>` deliberately leaves out."""
    base = set(_template_gpu_services("fleet/docker-compose.yml"))
    audio = set(_template_gpu_services("fleet/docker-compose.audio.yml"))
    assert set(_compose.GPU_SERVICES) <= base
    assert set(_compose.GPU_SERVICES_AUDIO) <= audio
    assert not set(_compose.GPU_SERVICES) & audio


# --- `lobes init` wiring ----------------------------------------------------


def _init(tmp_path, monkeypatch, *args, card: str = CSV_MODE_CARD):
    monkeypatch.setattr(_detect, "detect_card", lambda: _fake_card(card))
    return main(["init", str(tmp_path), *args])


def test_init_writes_both_halves_on_a_csv_card(tmp_path, monkeypatch, capsys) -> None:
    assert _init(tmp_path, monkeypatch, "--profile", CSV_MODE_CARD, "--apply") == 0
    capsys.readouterr()
    for name in (_compose.GPU_OVERLAY, _compose.GPU_AUDIO_OVERLAY):
        doc = _load_gpu_override(tmp_path, name)
        assert doc["services"], f"{name} declares no service"
    assert _compose.gpu_overlay_present(tmp_path)


def test_init_writes_nothing_on_a_default_card(tmp_path, monkeypatch, capsys) -> None:
    assert _init(tmp_path, monkeypatch, "--profile", "spark", "--apply", card="spark") == 0
    capsys.readouterr()
    assert not (tmp_path / _compose.GPU_OVERLAY).exists()
    assert not (tmp_path / _compose.GPU_AUDIO_OVERLAY).exists()
    assert not _compose.gpu_overlay_present(tmp_path)


def test_re_render_restores_the_csv_override(tmp_path, monkeypatch, capsys) -> None:
    """The acceptance criterion: the hand edit did not survive a re-init. This
    does — a second `--apply` rewrites a deleted/clobbered override."""
    _init(tmp_path, monkeypatch, "--profile", CSV_MODE_CARD, "--apply")
    (tmp_path / _compose.GPU_OVERLAY).unlink()
    (tmp_path / _compose.GPU_AUDIO_OVERLAY).write_text("services: {}\n", encoding="utf-8")
    _init(tmp_path, monkeypatch, "--profile", CSV_MODE_CARD, "--force", "--apply")
    capsys.readouterr()
    for name in (_compose.GPU_OVERLAY, _compose.GPU_AUDIO_OVERLAY):
        assert _load_gpu_override(tmp_path, name)["services"]


def test_re_render_onto_a_default_card_scrubs_the_override(tmp_path, monkeypatch, capsys) -> None:
    """The mirror image of the bug: a deployment moved onto a non-csv card must
    stop asking for the GPU the legacy way."""
    _init(tmp_path, monkeypatch, "--profile", CSV_MODE_CARD, "--apply")
    _init(tmp_path, monkeypatch, "--profile", "thor", "--force", "--apply", card="thor")
    capsys.readouterr()
    assert not (tmp_path / _compose.GPU_OVERLAY).exists()
    assert not (tmp_path / _compose.GPU_AUDIO_OVERLAY).exists()


def test_dry_run_plans_the_write(tmp_path, monkeypatch, capsys) -> None:
    assert _init(tmp_path, monkeypatch, "--profile", CSV_MODE_CARD, "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["gpu_override"] == {
        "files": [_compose.GPU_OVERLAY, _compose.GPU_AUDIO_OVERLAY],
        "action": "write",
        "gpu_access": GPU_ACCESS_RUNTIME,
    }
    assert not (tmp_path / _compose.GPU_OVERLAY).exists(), "dry run must write nothing"


def test_dry_run_plans_the_scrub(tmp_path, monkeypatch, capsys) -> None:
    _init(tmp_path, monkeypatch, "--profile", CSV_MODE_CARD, "--apply")
    capsys.readouterr()
    assert _init(tmp_path, monkeypatch, "--profile", "thor", "--json", card="thor") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["gpu_override"]["action"] == "remove"
    assert payload["gpu_override"]["gpu_access"] == GPU_ACCESS_DEVICES


def test_dry_run_is_silent_for_a_default_card(tmp_path, monkeypatch, capsys) -> None:
    assert _init(tmp_path, monkeypatch, "--profile", "spark", card="spark") == 0
    out = capsys.readouterr().out
    assert _compose.GPU_OVERLAY not in out
    assert _init(tmp_path, monkeypatch, "--profile", "spark", "--json", card="spark") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["gpu_override"] == {
        "files": [],
        "action": "none",
        "gpu_access": GPU_ACCESS_DEVICES,
    }


def test_apply_payload_records_what_landed(tmp_path, monkeypatch, capsys) -> None:
    assert _init(tmp_path, monkeypatch, "--profile", CSV_MODE_CARD, "--apply", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["gpu_override"] == {
        "files": [_compose.GPU_OVERLAY, _compose.GPU_AUDIO_OVERLAY],
        "written": True,
        "gpu_access": GPU_ACCESS_RUNTIME,
    }


def test_env_projection_is_untouched_by_gpu_access(tmp_path, monkeypatch, capsys) -> None:
    """gpu_access renders NO .env key — it selects a compose file, not a value —
    so the profile/shape goldens stay byte-identical on every card, orin
    included."""
    assert _init(tmp_path, monkeypatch, "--profile", CSV_MODE_CARD, "--apply", "--json") == 0
    capsys.readouterr()
    env_text = (tmp_path / _compose.ENV_FILE).read_text(encoding="utf-8")
    assert "gpu_access" not in env_text.lower()
    assert "GPU_ACCESS" not in env_text
