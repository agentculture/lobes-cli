"""Every env key ``lobes/realtime/_settings.py`` reads must be wired through
the deployment (issue #151 t8, restating the #149 s4 lesson: "a settings key
the compose env never passes is a dead knob").

Unlike ``test_fleet_audio_vad_env.py`` (a hand-curated ``_EXPECTED_DEFAULTS``
dict pinned to the six #149 VAD/AEC keys), this module derives the "keys
``build_settings`` reads" set STATICALLY from ``_settings.py``'s own source
via ``ast`` — no hand-maintained allowlist to forget updating when a new
field lands. A future ``Settings`` field read via ``env.get("KEY")`` /
``_as_int(env, "KEY", ...)`` / ``_as_float(env, "KEY", ...)`` is picked up
automatically; if nobody threads it through
``docker-compose.audio.yml``/``env.audio.example``, this test fails CI
without anyone having to remember to extend a list here.

Two DIFFERENT coverage rules, both keyed off compose itself rather than a
hand-picked exemption set:

1. **Every key must reach the container at all** — appear as SOME entry
   (literal or ``${...}``-interpolated) in the ``realtime`` service's
   ``environment:`` block. Without this the container falls back to
   ``build_settings``'s in-code default and no deployment change can move it
   (the exact #149 s4 bug: ``BARGE_IN_WINDOW_MS``/``BARGE_IN_MODEL`` were
   read by ``_settings.py`` since #149 and passed by nothing).
2. **Every key an OPERATOR can actually tune must be documented** —
   ``env.audio.example`` must carry a matching default, but only for keys
   compose exposes in the self-referencing ``${KEY:-default}`` form. A few
   keys (``TTS_URL``, ``STT_URL``, ``OPENAI_BASE_URL``, ``REALTIME_HOST``)
   are compose-network topology: their compose value never references
   ``${KEY...}`` for their OWN name (``STT_URL`` is computed from
   ``PARAKEET_PORT``, which IS a documented, tunable key; ``REALTIME_HOST``
   is hardcoded ``0.0.0.0``, the bind-all-inside-the-container convention
   shared with Chatterbox/Parakeet). Setting these directly in ``.env``
   would silently do nothing, so documenting them in ``env.audio.example``
   would be actively misleading — rule 2 is derived mechanically from
   compose's own value string, not from a list someone has to keep in sync.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import yaml

from lobes.realtime import _settings

_TEMPLATES = Path(__file__).resolve().parents[1] / "lobes" / "templates" / "fleet"
_AUDIO_COMPOSE = _TEMPLATES / "docker-compose.audio.yml"
_AUDIO_ENV_EXAMPLE = _TEMPLATES / "env.audio.example"


def _env_keys_read_by_build_settings() -> set[str]:
    """Every literal env-var key name ``build_settings()`` reads, via AST.

    Walks ``env.get("KEY")`` / ``_as_int(env, "KEY", default)`` /
    ``_as_float(env, "KEY", default)`` call shapes inside the function body.
    """
    source = inspect.getsource(_settings.build_settings)
    # inspect.getsource on a module-level function returns source starting at
    # column 0 (no leading indent to strip) — ast.parse needs no dedent here.
    tree = ast.parse(source)
    keys: set[str] = set()

    class _Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            else:
                name = None
            args = node.args
            if name == "get" and args and isinstance(args[0], ast.Constant):
                if isinstance(args[0].value, str):
                    keys.add(args[0].value)
            elif name in ("_as_int", "_as_float") and len(args) >= 2:
                key_arg = args[1]
                if isinstance(key_arg, ast.Constant) and isinstance(key_arg.value, str):
                    keys.add(key_arg.value)
            self.generic_visit(node)

    _Visitor().visit(tree)
    return keys


def _realtime_env_map() -> dict[str, str]:
    """The ``realtime`` service's ``environment:`` list as ``{KEY: raw-value}``."""
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


def _is_self_referencing(key: str, value: str) -> bool:
    """True iff *value* is compose's ``${KEY:-default}``/``${KEY}`` pattern
    for THIS key — the mechanical signal that an operator's own ``.env``
    setting actually reaches the container (vs. a hardcoded/derived value a
    ``.env`` override would silently not affect)."""
    return bool(re.fullmatch(rf"\$\{{{re.escape(key)}(:-[^}}]*)?\}}", value))


class TestEveryReadKeyReachesTheContainer:
    """Rule 1: every key ``build_settings`` reads must appear as SOME entry
    in the ``realtime`` service's ``environment:`` block — the #149 s4 bug
    class (a key read but never passed silently pins to the code default)."""

    def test_no_settings_key_is_unwired(self) -> None:
        read_keys = _env_keys_read_by_build_settings()
        compose_keys = set(_realtime_env_map())
        missing = sorted(read_keys - compose_keys)
        assert not missing, (
            f"lobes/realtime/_settings.py reads {missing} but "
            "docker-compose.audio.yml's realtime service never passes them — "
            "a dead knob, permanently pinned to its in-code default "
            "(the #149 s4 lesson)"
        )


class TestEveryOperatorTunableKeyIsDocumented:
    """Rule 2: every key compose exposes as operator-overridable
    (``${KEY:-default}``) must also appear in ``env.audio.example`` with a
    matching default — the doctor-heal source of truth."""

    def test_no_tunable_key_is_undocumented(self) -> None:
        read_keys = _env_keys_read_by_build_settings()
        compose = _realtime_env_map()
        tunable = {k for k in read_keys if k in compose and _is_self_referencing(k, compose[k])}
        example_keys = set(_env_example_defaults())
        missing = sorted(tunable - example_keys)
        assert not missing, (
            f"{missing} are operator-tunable in docker-compose.audio.yml "
            "(${KEY:-default}) but env.audio.example does not document "
            "them — doctor --fix has no default to heal them with"
        )

    def test_documented_defaults_match_compose_defaults(self) -> None:
        """A drift here means the deployed default and the doctor-heal /
        documented default disagree — exactly the class of silent bug this
        task exists to close."""
        read_keys = _env_keys_read_by_build_settings()
        compose = _realtime_env_map()
        example = _env_example_defaults()
        for key in sorted(read_keys):
            value = compose.get(key, "")
            if not _is_self_referencing(key, value):
                continue
            match = re.fullmatch(rf"\$\{{{re.escape(key)}:-([^}}]*)\}}", value)
            compose_default = match.group(1) if match else ""
            assert key in example, f"{key} missing from env.audio.example"
            assert example[key] == compose_default, (
                f"{key} default drifted: docker-compose.audio.yml has "
                f"{compose_default!r}, env.audio.example has {example[key]!r}"
            )


class TestTheFourNewOrPreviouslyDeadKeys:
    """Spot-checks naming the specific keys this task wires, so a reviewer
    (or a future git-blame reader) sees the exact list without having to
    reconstruct it from the general rules above."""

    _EXPECTED = {
        "BARGE_IN_WINDOW_MS": "750",
        "BARGE_IN_MODEL": "",
        "TTS_VOICE_CONCURRENCY": "1",
        "TTS_CONCURRENCY": "1",
        "DEFAULT_SYSTEM_PROMPT": "",
    }

    def test_present_in_compose_and_example_with_expected_default(self) -> None:
        compose = _realtime_env_map()
        example = _env_example_defaults()
        for key, default in self._EXPECTED.items():
            assert key in compose, f"{key} missing from docker-compose.audio.yml"
            assert compose[key] == f"${{{key}:-{default}}}", (
                f"{key} in compose is {compose[key]!r}, expected the "
                f"operator-overridable ${{{key}:-{default}}} form"
            )
            assert key in example, f"{key} missing from env.audio.example"
            assert (
                example[key] == default
            ), f"{key} in env.audio.example is {example[key]!r}, expected {default!r}"

    def test_barge_in_and_tts_voice_concurrency_match_settings_defaults(self) -> None:
        s = _settings.build_settings({})
        assert s.barge_in_window_ms == int(self._EXPECTED["BARGE_IN_WINDOW_MS"])
        assert s.barge_in_model is None  # empty env -> None, not ""
        assert s.tts_voice_concurrency == int(self._EXPECTED["TTS_VOICE_CONCURRENCY"])
        assert s.tts_concurrency == int(self._EXPECTED["TTS_CONCURRENCY"])
        # DEFAULT_SYSTEM_PROMPT's *compose*/env.audio.example default is the
        # empty string, but build_settings mirrors _session.DEFAULT_SYSTEM_PROMPT
        # when the env value is empty — never a blank prompt in practice.
        assert s.default_system_prompt == _settings._SESSION_DEFAULT_SYSTEM_PROMPT
        assert s.default_system_prompt != ""
