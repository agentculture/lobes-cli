"""The audio overlay's realtime service must pass the VAD / turn-detection /
AEC knobs that ``lobes/realtime/_settings.py`` already reads (issue #149).

Before this test's target lands, grep finds NEITHER
``docker-compose.audio.yml``'s ``realtime`` service NOR ``env.audio.example``
naming ``VAD_THRESHOLD`` / ``VAD_SILENCE_MS`` / ``VAD_PREFIX_PADDING_MS`` /
``DEFAULT_TURN_DETECTION`` / ``DEFAULT_AEC_MODE`` / ``VAD_MAX_TURN_MS`` — so a
deployed container silently pins to ``build_settings``'s code defaults and no
operator can tune them. This locks two things: (1) every key reaches the
``realtime`` service's ``environment:`` block with the SAME default
``build_settings`` uses (a drift between compose and code is a bug), and (2)
``env.audio.example`` documents each key with that default and a comment.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from lobes.realtime._settings import build_settings

_TEMPLATES = Path(__file__).resolve().parents[1] / "lobes" / "templates" / "fleet"
_AUDIO_COMPOSE = _TEMPLATES / "docker-compose.audio.yml"
_AUDIO_ENV_EXAMPLE = _TEMPLATES / "env.audio.example"

# The exact keys this task wires, and the default each must carry — pinned to
# lobes/realtime/_settings.py::build_settings's own defaults (the single
# source of truth; VAD_MAX_TURN_MS is new — t1/t6 wire the settings side, this
# task only carries the env name + the same 30000 ms default through).
_EXPECTED_DEFAULTS = {
    "VAD_THRESHOLD": "0.5",
    "VAD_SILENCE_MS": "600",
    "VAD_PREFIX_PADDING_MS": "300",
    "VAD_MAX_TURN_MS": "30000",
    "DEFAULT_TURN_DETECTION": "server_vad",
    "DEFAULT_AEC_MODE": "none",
}


def _realtime_env_map() -> dict[str, str]:
    """The ``realtime`` service's ``environment:`` list as ``{KEY: raw-value}``.

    Values keep their raw ``${...}`` interpolation text (PyYAML expands
    nothing).
    """
    compose = yaml.safe_load(_AUDIO_COMPOSE.read_text(encoding="utf-8"))
    env: list[str] = compose["services"]["realtime"]["environment"]
    out: dict[str, str] = {}
    for entry in env:
        assert "=" in entry, f"non key=value realtime environment entry: {entry!r}"
        key, _, value = entry.partition("=")
        out[key] = value
    return out


def _env_example_defaults() -> dict[str, str]:
    """``KEY=VALUE`` lines from ``env.audio.example``, keyed by env var."""
    out: dict[str, str] = {}
    for line in _AUDIO_ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        out[key.strip()] = value.strip()
    return out


class TestRealtimeServiceReceivesEveryVadKey:
    """Every key in the contract must reach the ``realtime`` service."""

    def test_all_keys_present(self) -> None:
        env = _realtime_env_map()
        missing = sorted(set(_EXPECTED_DEFAULTS) - set(env))
        assert not missing, f"realtime service is missing env key(s): {missing}"

    def test_defaults_match_build_settings(self) -> None:
        """The compose ``${KEY:-default}`` value must equal build_settings's own
        default — a drift between compose and code is a bug."""
        env = _realtime_env_map()
        for key, default in _EXPECTED_DEFAULTS.items():
            expected = f"${{{key}:-{default}}}"
            assert env[key] == expected, (
                f"{key} default drifted: compose has {env[key]!r}, expected "
                f"{expected!r} to match lobes/realtime/_settings.py"
            )

    def test_still_operator_overridable(self) -> None:
        env = _realtime_env_map()
        for key in _EXPECTED_DEFAULTS:
            assert env[key].startswith(f"${{{key}:-"), f"{key} must stay operator-overridable"


class TestEnvAudioExampleDocumentsEveryVadKey:
    """``env.audio.example`` is what ``append_audio_env`` appends to a fresh
    scaffold's ``.env`` — every key must be present, at the SAME default, with
    a documenting comment."""

    def test_all_keys_present_with_matching_default(self) -> None:
        defaults = _env_example_defaults()
        for key, default in _EXPECTED_DEFAULTS.items():
            assert key in defaults, f"env.audio.example is missing {key}"
            assert defaults[key] == default, (
                f"{key} in env.audio.example is {defaults[key]!r}, expected {default!r} "
                "to match lobes/realtime/_settings.py"
            )

    def test_each_key_has_a_documenting_comment(self) -> None:
        text = _AUDIO_ENV_EXAMPLE.read_text(encoding="utf-8")
        lines = text.splitlines()
        for key in _EXPECTED_DEFAULTS:
            idx = next(i for i, ln in enumerate(lines) if ln.startswith(f"{key}="))
            # A comment line documenting this key sits somewhere above it,
            # before any other KEY= assignment or blank-line paragraph break.
            found = False
            for j in range(idx - 1, -1, -1):
                candidate = lines[j]
                if candidate.strip() == "":
                    break
                if candidate.lstrip().startswith("#"):
                    found = True
                    break
                if "=" in candidate:
                    break
            assert found, f"{key} has no documenting comment directly above it"


class TestSettingsDefaultsMatchTheContract:
    """Belt-and-suspenders: build_settings's own defaults are what this test's
    expectations are pinned to — assert that pin stays true."""

    def test_build_settings_defaults(self) -> None:
        s = build_settings({})
        assert s.vad_threshold == float(_EXPECTED_DEFAULTS["VAD_THRESHOLD"])
        assert s.vad_silence_ms == int(_EXPECTED_DEFAULTS["VAD_SILENCE_MS"])
        assert s.vad_prefix_padding_ms == int(_EXPECTED_DEFAULTS["VAD_PREFIX_PADDING_MS"])
        assert s.default_turn_detection == _EXPECTED_DEFAULTS["DEFAULT_TURN_DETECTION"]
        assert s.default_aec_mode == _EXPECTED_DEFAULTS["DEFAULT_AEC_MODE"]
