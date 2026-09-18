"""``GENERATE_STREAM`` — the one knob that turns sentence-level streaming off.

Default-ON (approved deviation d7, 2026-09-18), because streaming is the
measured behaviour this deployment ships. That direction is why the parser is
a DENYLIST of falsy tokens rather than an allowlist of truthy ones: a typo
must leave the fast path armed, not silently revert a deployment to the slow
one nobody asked for.
"""

from __future__ import annotations

import pytest

from lobes.realtime._settings import build_settings


def test_an_unset_knob_streams() -> None:
    assert build_settings({}).generate_stream is True


@pytest.mark.parametrize("value", ("0", "false", "FALSE", "No", " off "))
def test_an_explicit_falsy_token_turns_streaming_off(value: str) -> None:
    assert build_settings({"GENERATE_STREAM": value}).generate_stream is False


@pytest.mark.parametrize("value", ("1", "true", "yes", "on", "ture", ""))
def test_anything_else_leaves_streaming_on(value: str) -> None:
    # Including a typo: the default-on direction makes that the safe answer.
    assert build_settings({"GENERATE_STREAM": value}).generate_stream is True


def test_the_hebrew_overlay_is_the_one_that_wires_it() -> None:
    # The English overlay stays byte-identical (the Hebrew overlay's own
    # criterion 2), so the key is declared and documented only there.
    from pathlib import Path

    templates = Path(__file__).resolve().parents[1] / "lobes" / "templates" / "fleet"
    assert "GENERATE_STREAM" in (templates / "docker-compose.audio-he.yml").read_text(
        encoding="utf-8"
    )
    assert "GENERATE_STREAM" in (templates / "env.audio-he.example").read_text(encoding="utf-8")
    assert "GENERATE_STREAM" not in (templates / "docker-compose.audio.yml").read_text(
        encoding="utf-8"
    )
    assert "GENERATE_STREAM" not in (templates / "env.audio.example").read_text(encoding="utf-8")
