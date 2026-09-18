"""Tests for the stt/tts capabilities advert naming the SERVED engine
(issue #204, task t13, spec claim c27).

Before this task ``lobes.roles`` hardcoded ``_STT_MODEL`` /
``_TTS_MODEL`` (Parakeet / Chatterbox) unconditionally — a Hebrew (or any
non-English) deployment serving a different STT/TTS pair would advertise
an English engine it does not run. :func:`lobes.roles.build_role_registry`
now reads a deployment's own ``STT_MODEL``/``STT_RUNTIME`` (and the TTS
equivalents) from ``env`` and falls back to the historical Parakeet/
Chatterbox constants when nothing is declared — see
:func:`lobes.roles._declared_audio_engine`.

``language`` is deliberately OUT of scope here: honoring it would need
``RoleInfo`` to gain a field that is ABSENT from the JSON advert when
unset, and both places that render the advert
(``lobes/cli/_commands/capabilities.py:_role_payload`` and
``lobes/gateway/server.py:capabilities_payload``) serialise ``RoleInfo``
with a blind ``dataclasses.asdict(info)`` that always emits every field —
a change this task is not permitted to make (see the module-level comment
above ``_STT_MODEL_ENV`` in ``lobes/roles.py``, and t13's own stop
condition). This module therefore pins that ``RoleInfo`` carries no
``language`` field today, as a guard against silently reintroducing one
without also fixing the two serialisation call sites.
"""

from __future__ import annotations

import dataclasses

from lobes.gateway._config import build_config
from lobes.roles import RoleInfo, build_role_registry, role_payload

_PRIMARY_ID = "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"

_FULL_ENV = {
    "PRIMARY_URL": "http://vllm-primary:8000",
    "PRIMARY_SERVED_NAME": _PRIMARY_ID,
    "AUDIO_URL": "http://realtime:8080",
}


def _registry(env: dict[str, str]) -> dict[str, RoleInfo]:
    table, server = build_config(env)
    return build_role_registry(table, server, env=env, gateway_url="http://localhost:8000")


# ---------------------------------------------------------------------------
# Pin today's (pre-declaration) behavior — the "byte-identical to main"
# acceptance criterion, captured from the CURRENT code before any change.
# ---------------------------------------------------------------------------


def test_nothing_declared_stt_advert_matches_main() -> None:
    registry = _registry(dict(_FULL_ENV))
    stt = registry["stt"]
    assert stt.model == "nvidia/parakeet-tdt-0.6b-v2"
    assert stt.runtime == "parakeet"


def test_nothing_declared_tts_advert_matches_main() -> None:
    registry = _registry(dict(_FULL_ENV))
    tts = registry["tts"]
    assert tts.model == "ResembleAI/chatterbox"
    assert tts.runtime == "chatterbox"


def test_nothing_declared_advert_full_field_snapshot_matches_main() -> None:
    """Pins the ENTIRE serialised stt/tts payload (dataclasses.asdict), not
    just model/runtime, against exactly what main produces today — captured
    from the pre-change code so a future edit cannot silently widen or
    narrow the field set or reorder keys."""
    registry = _registry(dict(_FULL_ENV))
    stt_payload = role_payload(registry["stt"])
    tts_payload = role_payload(registry["tts"])
    assert list(stt_payload.keys()) == [
        "role",
        "model",
        "runtime",
        "endpoint",
        "path",
        "context",
        "quant",
        "mtp",
        "tools",
        "responsibilities",
        "forbidden_responsibilities",
        "feasible",
        "ready",
        "loaded",
    ]
    assert stt_payload["model"] == "nvidia/parakeet-tdt-0.6b-v2"
    assert stt_payload["runtime"] == "parakeet"
    assert tts_payload["model"] == "ResembleAI/chatterbox"
    assert tts_payload["runtime"] == "chatterbox"
    assert list(tts_payload.keys()) == list(stt_payload.keys())


# ---------------------------------------------------------------------------
# Declared overrides — the new behavior (c27)
# ---------------------------------------------------------------------------


def test_stt_model_and_runtime_declared_via_env() -> None:
    env = dict(_FULL_ENV)
    env["STT_MODEL"] = "ivrit-ai/whisper-large-v3-turbo-ct2"
    env["STT_RUNTIME"] = "whisper"
    registry = _registry(env)
    assert registry["stt"].model == "ivrit-ai/whisper-large-v3-turbo-ct2"
    assert registry["stt"].runtime == "whisper"
    # tts stays on the untouched default — declaring stt does not leak.
    assert registry["tts"].model == "ResembleAI/chatterbox"
    assert registry["tts"].runtime == "chatterbox"


def test_tts_model_and_runtime_declared_via_env() -> None:
    env = dict(_FULL_ENV)
    env["TTS_MODEL"] = "some-org/hebrew-multilingual-tts"
    env["TTS_RUNTIME"] = "multilingual-tts"
    registry = _registry(env)
    assert registry["tts"].model == "some-org/hebrew-multilingual-tts"
    assert registry["tts"].runtime == "multilingual-tts"
    # stt stays on the untouched default.
    assert registry["stt"].model == "nvidia/parakeet-tdt-0.6b-v2"
    assert registry["stt"].runtime == "parakeet"


def test_declaring_model_alone_leaves_runtime_on_default() -> None:
    """Declaring only STT_MODEL overrides just the model — the runtime keeps
    its historical default rather than going blank."""
    env = dict(_FULL_ENV)
    env["STT_MODEL"] = "ivrit-ai/whisper-large-v3-turbo-ct2"
    registry = _registry(env)
    assert registry["stt"].model == "ivrit-ai/whisper-large-v3-turbo-ct2"
    assert registry["stt"].runtime == "parakeet"


def test_declaring_runtime_alone_leaves_model_on_default() -> None:
    env = dict(_FULL_ENV)
    env["TTS_RUNTIME"] = "multilingual-tts"
    registry = _registry(env)
    assert registry["tts"].model == "ResembleAI/chatterbox"
    assert registry["tts"].runtime == "multilingual-tts"


def test_blank_declared_values_fall_back_to_default() -> None:
    """A present-but-blank env value (e.g. an unset compose passthrough that
    still sets the key to "") must not win over the default — matches the
    blank-value discipline every other env-read helper in this module uses
    (_declared_context, _served_context)."""
    env = dict(_FULL_ENV)
    env["STT_MODEL"] = ""
    env["STT_RUNTIME"] = "   "
    registry = _registry(env)
    assert registry["stt"].model == "nvidia/parakeet-tdt-0.6b-v2"
    assert registry["stt"].runtime == "parakeet"


def test_declared_override_survives_infeasible_lane() -> None:
    """A declared-off lane (STT_FEASIBLE=false) still advertises the
    declared model/runtime for what it WOULD serve, mirroring the existing
    "unloaded role still names the model it would serve" contract for every
    other role."""
    env = dict(_FULL_ENV)
    env["STT_MODEL"] = "ivrit-ai/whisper-large-v3-turbo-ct2"
    env["STT_FEASIBLE"] = "false"
    registry = _registry(env)
    stt = registry["stt"]
    assert stt.feasible is False
    assert stt.model == "ivrit-ai/whisper-large-v3-turbo-ct2"


# ---------------------------------------------------------------------------
# language: the guard that stood here ("RoleInfo carries no language field
# until both asdict call sites are fixed") was REDEEMED by approved deviation
# d3 — both call sites now go through lobes.roles.role_payload, which omits the
# key when unset. The behaviour is pinned in tests/test_roles_audio_language.py.
# ---------------------------------------------------------------------------


def test_the_language_field_exists_only_with_the_omitting_serializer() -> None:

    field_names = {f.name for f in dataclasses.fields(RoleInfo)}
    assert "language" in field_names
    assert "language" not in role_payload(_registry(dict(_FULL_ENV))["stt"])
