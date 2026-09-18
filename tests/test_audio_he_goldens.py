"""Golden for the Hebrew overlay's ``${VAR:-default}`` substitution surface (t15).

``tests/goldens/template-defaults.env`` pins the BASE fleet compose's defaults
— the surface a deployment runs on when a knob is unresolved. The Hebrew
overlay adds a whole second such surface (``STT_MODEL``, ``TTS_TEMPERATURE``,
``REALTIME_LANGUAGE``, the continuation/eager-VAD knobs …), and nothing in
``tests/goldens/`` could see an edit to it.

The golden lives under ``tests/goldens/overlays/`` rather than beside
``template-defaults.env`` deliberately: ``tests/test_profile_goldens.py``
asserts the top-level ``*.env`` set equals the built-in PROFILE set, and an
overlay is not a profile.
"""

from __future__ import annotations

from pathlib import Path

from tests.goldens.regen import AUDIO_HE_COMPOSE, audio_he_defaults_text

_GOLDEN = Path(__file__).resolve().parent / "goldens" / "overlays" / "audio-he-defaults.env"
_REGEN_CMD = "uv run python tests/goldens/regen.py"


def test_audio_he_defaults_golden_byte_for_byte() -> None:
    actual = audio_he_defaults_text()
    expected = _GOLDEN.read_text(encoding="utf-8")
    assert actual == expected, (
        f"tests/goldens/overlays/audio-he-defaults.env drifted from the "
        f"${{VAR:-default}} surface of {AUDIO_HE_COMPOSE}.\n"
        f"If this is a deliberate change, regenerate with: {_REGEN_CMD}"
    )


def test_audio_he_golden_carries_the_hebrew_selection() -> None:
    """A spot-check that the golden is the HEBREW surface and not a copy of the
    English one — the two overlays share key names, not values."""
    text = _GOLDEN.read_text(encoding="utf-8")
    assert "REALTIME_LANGUAGE=he" in text
    assert "STT_LANGUAGE=he" in text
    assert "TTS_LANGUAGE=he" in text


def test_audio_he_golden_is_pure() -> None:
    """Rendering is a pure function of the template — no host state."""
    first = audio_he_defaults_text()
    second = audio_he_defaults_text()
    assert first == second
    assert first == _GOLDEN.read_text(encoding="utf-8")
