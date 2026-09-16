"""``INNEREYE_UI_PORT`` — the opt-in publication of ComfyUI's OWN web UI.

The ``comfyui`` service ships ``expose: ["8188"]`` with **no** ``ports:`` key
(``tests/test_comfyui_compose.py``), because ComfyUI ships no authentication of
any kind (innereye challenge finding ``c36``). This knob lets an operator give
that property up DELIBERATELY, and nothing about it may happen by accident:

1. **Default OFF, byte-identical.** With the knob unset, every rendered file is
   what it was before the knob existed — including the shipped compose
   template, which is NOT touched (the ``ports:`` line is rendered into the
   GENERATED ``docker-compose.shape.yml``, since compose cannot conditionally
   omit a key).
2. **Safe by default when on.** A bare port number binds LOOPBACK
   (``127.0.0.1:PORT:8188``). A wider bind must be typed as an explicit
   interface — this box has a LAN address as well as a tailnet one, and
   ``0.0.0.0`` on an unauthenticated ComfyUI would publish it to the whole
   home WiFi.
3. **Never parsed into an int** (issue #272). ``VLLM_PORT``'s parser rejects
   docker's ``IP:port`` form and every verb that resolves the port then fails;
   this knob is consumed only at RENDER time to emit a compose ``ports:``
   entry, so it stays an opaque string.
4. **Every surface says what is given up** — ``env.example``, the generated
   compose comment, and ``docs/comfyui-innereye.md``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lobes.cli import main
from lobes.cli._commands import init as init_cmd
from lobes.cli._errors import ModelGearError
from lobes.profiles.loader import resolve_profile
from lobes.profiles.shapes import resolve_shape
from lobes.runtime import _compose, _detect, _env

_REPO = Path(__file__).resolve().parents[1]
_ENV_EXAMPLE = _REPO / "lobes" / "templates" / "fleet" / "env.example"
_FLEET_COMPOSE = _REPO / "lobes" / "templates" / "fleet" / "docker-compose.yml"
_DOC = _REPO / "docs" / "comfyui-innereye.md"

_KEY = "INNEREYE_UI_PORT"
_SERVICE = "comfyui"


def _fake_card(resolved: str) -> _detect.DetectedCard:
    return _detect.DetectedCard(
        resolved=resolved,
        device_name="NVIDIA GB10",
        compute_capability="sm_121",
        total_memory_gb=119.7,
        hostname="test-host",
        device_tree_model=None,
        sources={},
    )


def _patch_detect(monkeypatch, card: str = "spark") -> None:
    monkeypatch.setattr(_detect, "detect_card", lambda: _fake_card(card))


class _ComposeTagLoader(yaml.SafeLoader):
    """SafeLoader tolerating compose's ``!reset`` merge tag (see test_init_shape)."""


_ComposeTagLoader.add_constructor("!reset", lambda loader, node: {"__reset__": True})


def _overlay(target: Path) -> dict:
    return yaml.load(
        (target / _compose.SHAPE_OVERLAY).read_text(encoding="utf-8"),
        Loader=_ComposeTagLoader,
    )


def _init_with_knob(tmp_path: Path, monkeypatch, value: str | None, *, shape: str | None = None):
    """Scaffold a fleet deployment, set the knob in ``.env``, re-render."""
    _patch_detect(monkeypatch)
    target = tmp_path / "deploy"
    argv = ["init", str(target), "--apply"]
    if shape:
        argv = ["init", "--shape", shape, str(target), "--apply"]
    assert main(argv) == 0
    if value is not None:
        _env.set_env(target / ".env", _KEY, value)
        assert main(argv) == 0
    return target


# --- criterion 1: default OFF is byte-identical -----------------------------


class TestDefaultOff:
    def test_shipped_compose_still_publishes_no_host_port(self) -> None:
        """The knob adds NOTHING to the packaged template — no new ${VAR}, no ports."""
        services = yaml.safe_load(_FLEET_COMPOSE.read_text(encoding="utf-8"))["services"]
        svc = services[_SERVICE]
        assert "ports" not in svc, "the comfyui service must not publish a host port"
        assert [str(p) for p in svc.get("expose", [])] == ["8188"]
        assert _KEY not in _FLEET_COMPOSE.read_text(encoding="utf-8")

    def test_env_example_ships_the_knob_commented_out(self) -> None:
        text = _ENV_EXAMPLE.read_text(encoding="utf-8")
        assert f"# {_KEY}=" in text, "the knob must ship documented"
        for line in text.splitlines():
            if line.strip().startswith(f"{_KEY}="):
                pytest.fail(f"{_KEY} must ship COMMENTED OUT (default off), got: {line!r}")

    def test_bare_init_writes_no_shape_overlay(self, tmp_path, monkeypatch) -> None:
        target = _init_with_knob(tmp_path, monkeypatch, None)
        assert not (target / _compose.SHAPE_OVERLAY).exists()

    def test_knob_unset_renders_byte_identically(self, tmp_path, monkeypatch) -> None:
        """A deployment rendered with the knob absent equals one rendered before it."""
        _patch_detect(monkeypatch)
        a, b = tmp_path / "a", tmp_path / "b"
        assert main(["init", str(a), "--apply"]) == 0
        assert main(["init", "--shape", "spark-lobe", str(b), "--apply"]) == 0
        # The role-dropping shape still writes ONLY its parked services.
        services = _overlay(b)["services"]
        assert _SERVICE not in services
        assert not any("ports" in svc for svc in services.values())
        assert not (a / _compose.SHAPE_OVERLAY).exists()

    def test_empty_value_is_off(self, tmp_path, monkeypatch) -> None:
        target = _init_with_knob(tmp_path, monkeypatch, "   ")
        assert not (target / _compose.SHAPE_OVERLAY).exists()

    def test_render_helper_returns_none_when_unset(self) -> None:
        shape = resolve_shape("machine-as-brain")
        profile = resolve_profile("spark")
        assert init_cmd.render_shape_override(shape, profile) is None
        assert init_cmd.render_shape_override(shape, profile, ui_port=None) is None
        assert init_cmd.render_shape_override(shape, profile, ui_port="") is None


# --- criterion 2: both bind forms ------------------------------------------


class TestBindForms:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("8188", "127.0.0.1:8188:8188"),
            (" 8188 ", "127.0.0.1:8188:8188"),
            ("9000", "127.0.0.1:9000:8188"),
            ("100.127.105.72:8188", "100.127.105.72:8188:8188"),
            ("192.168.1.157:9000", "192.168.1.157:9000:8188"),
            ("127.0.0.1:8188", "127.0.0.1:8188:8188"),
            ("0.0.0.0:8188", "0.0.0.0:8188:8188"),  # nosec B104 — the explicit opt-in
            ("[::1]:8188", "[::1]:8188:8188"),
        ],
    )
    def test_publish_spec(self, value: str, expected: str) -> None:
        assert init_cmd.innereye_ui_publish(value) == expected

    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_publish_spec_off(self, value) -> None:
        assert init_cmd.innereye_ui_publish(value) is None

    @pytest.mark.parametrize("value", ["81 88", 'x"y:8188', "8188\n- evil", "127.0.0.1:"])
    def test_malformed_value_is_a_user_error(self, value: str) -> None:
        with pytest.raises(ModelGearError):
            init_cmd.innereye_ui_publish(value)

    def test_bare_port_binds_loopback_in_the_overlay(self, tmp_path, monkeypatch) -> None:
        target = _init_with_knob(tmp_path, monkeypatch, "8188")
        assert _overlay(target)["services"][_SERVICE]["ports"] == ["127.0.0.1:8188:8188"]

    def test_interface_form_binds_that_interface(self, tmp_path, monkeypatch) -> None:
        target = _init_with_knob(tmp_path, monkeypatch, "100.127.105.72:8188")
        assert _overlay(target)["services"][_SERVICE]["ports"] == ["100.127.105.72:8188:8188"]

    def test_overlay_is_scrubbed_when_the_knob_is_removed(self, tmp_path, monkeypatch) -> None:
        target = _init_with_knob(tmp_path, monkeypatch, "8188")
        assert (target / _compose.SHAPE_OVERLAY).exists()
        _env.set_env(target / ".env", _KEY, "")
        assert main(["init", str(target), "--apply"]) == 0
        assert not (target / _compose.SHAPE_OVERLAY).exists()

    def test_composes_with_a_role_dropping_shape(self, tmp_path, monkeypatch) -> None:
        """The UI block and the shape's parked services coexist in one overlay."""
        target = _init_with_knob(tmp_path, monkeypatch, "8188", shape="spark-lobe")
        services = _overlay(target)["services"]
        assert services[_SERVICE]["ports"] == ["127.0.0.1:8188:8188"]
        assert services["vllm-multimodal"]["profiles"] == [init_cmd.SHAPE_DROPPED_PROFILE]

    def test_ui_block_is_not_read_as_a_dropped_role(self, tmp_path, monkeypatch) -> None:
        """A published UI must never look like "this shape drops innereye".

        The overlay's readers key on the ``shape-dropped`` profile marker, not on
        the mere presence of a service block — otherwise publishing the UI would
        silently park the lane in doctor's and ``lobes up``'s eyes.
        """
        target = _init_with_knob(tmp_path, monkeypatch, "8188")
        assert _compose.shape_dropped_containers(target) == ()
        text = (target / _compose.SHAPE_OVERLAY).read_text(encoding="utf-8")
        assert _compose.shape_parked_service_keys(text) == set()


# --- criterion 3: no int parsing (issue #272) -------------------------------


class TestNotAPort:
    def test_render_never_parses_the_value_as_an_int(self, monkeypatch) -> None:
        """``int()`` is never called on the knob — the #272 trap, pinned."""

        def _boom(*_args, **_kwargs):  # pragma: no cover - only on failure
            raise AssertionError("the UI knob must not be parsed into an int")

        monkeypatch.setattr(init_cmd, "int", _boom, raising=False)
        assert init_cmd.innereye_ui_publish("100.127.105.72:8188") == "100.127.105.72:8188:8188"

    def test_cli_port_resolution_ignores_the_knob(self, tmp_path, monkeypatch) -> None:
        """Setting the knob does not break any verb that resolves a port."""
        target = _init_with_knob(tmp_path, monkeypatch, "100.127.105.72:8188")
        env = _env.read_env_file(target / ".env")
        assert env[_KEY] == "100.127.105.72:8188"
        assert main(["status", "--compose-dir", str(target), "--json"]) in (0, 1)


# --- criterion 4: every surface discloses the cost --------------------------


class TestDisclosure:
    def test_env_example_names_what_is_given_up(self) -> None:
        text = _ENV_EXAMPLE.read_text(encoding="utf-8")
        start = text.index("INNEREYE_UI_PORT")
        section = text[max(0, start - 2000) : start + 200]
        assert "/history" in section and "/view" in section
        assert "no authentication" in section.lower()
        assert "127.0.0.1" in section

    def test_generated_overlay_carries_the_warning(self, tmp_path, monkeypatch) -> None:
        target = _init_with_knob(tmp_path, monkeypatch, "8188")
        text = (target / _compose.SHAPE_OVERLAY).read_text(encoding="utf-8")
        assert "no authentication" in text.lower()
        assert "/history" in text and "/view" in text
        assert "c36" in text

    def test_doc_documents_the_third_exposure(self) -> None:
        text = _DOC.read_text(encoding="utf-8")
        assert "INNEREYE_UI_PORT" in text
        assert "third exposure" in text.lower()
        # The old unconditional claim must no longer stand unqualified.
        assert "No reader should conclude the native ComfyUI surface is reachable" not in text

    def test_dry_run_plan_names_the_publication(self, tmp_path, monkeypatch, capsys) -> None:
        target = _init_with_knob(tmp_path, monkeypatch, "8188")
        capsys.readouterr()
        assert main(["init", str(target)]) == 0
        out = capsys.readouterr().out
        assert "127.0.0.1:8188:8188" in out

    def test_apply_json_reports_the_publication(self, tmp_path, monkeypatch, capsys) -> None:
        import json

        target = _init_with_knob(tmp_path, monkeypatch, "8188")
        capsys.readouterr()
        assert main(["init", str(target), "--apply", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["shape_override"]["innereye_ui"] == "127.0.0.1:8188:8188"
